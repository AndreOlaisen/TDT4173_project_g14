import os
import re
import string
import math
import argparse
import torch
import torch.nn as nn
import shared_model
from export_activations import load_run_stats
from fastai.learner import load_model
from pathlib import Path
from collections import Counter
from enum import Enum
import cmsis_nn.cfile.cfile as C

# default_base_path = Path(os.getenv("FASTAI_HOME", "~/.fastai")).expanduser()
static_in_shape = [3, 32, 32]
static_out_shape = [10]

"""
TODO:
    - Gather activation stats from validation data, use it to scale the output from conv/fc layers
    - Fold batchnorm into conv/fc layers
"""

############################################################################
# Model conversion 
############################################################################

class LayerType(Enum):
    INPUT = "INPUT"
    CONV_RGB = "CONV_RGB"
    CONV = "CONV"
    RELU = "RELU"
    MAXPOOL = "POOL"
    FLATTEN = "FLATTEN"
    FC = "FC"
    SOFTMAX = "SOFTMAX"


def reorder_conv2d(weight, bias):
    return weight.permute(0, 3, 1, 2), bias


def quantize(weights, databits, *args):
    min_wt_abs = abs(weights.min().item())
    max_wt_abs = abs(weights.max().item())
    ibits = max(int(math.ceil(math.log2(max(min_wt_abs, max_wt_abs)))), 0)
    fbits = databits - ibits
    quant_weight = (weights * (2**fbits)).round()
    return quant_weight, ibits, fbits


class Converter:
    def __init__(self, in_shape, out_shape, datatype, act_hists, bn_mean_vars, qwrange, use_opt=False):
        assert(len(in_shape) == 3 and in_shape[1] == in_shape[2])
        assert(len(out_shape) == 1)
        assert(use_opt == False)
        self.modules = []
        self.id_counter = Counter()
        self.in_shape = in_shape
        self.out_shape = out_shape
        if datatype == "q7_t":
            self.databits = 8 - 1  # 8 bits minus sign bit
        else:
            self.databits = 16 - 1  # 16 bits minus sign bit
        self.act_formats = self._calc_act_formats(act_hists)
        self.bn_mean_vars = bn_mean_vars
        self._convert_input()

    def _id_gen(self, layer_type):
        """ Generate unique ID number for layer with type layer_type """ 
        count = self.id_counter[layer_type]
        self.id_counter[layer_type] += 1
        return count

    def _calc_act_formats(self, act_hists):
        formats = []
        for hist in act_hists:
            for i, val in enumerate(hist[:self.databits]): 
                if val == 0:
                    formats.append((i, self.databits - i))
                    break
            else:
                formats.append((self.databits, 0))
                # TODO: warning?
        return formats

    def _convert_input(self):
        # TODO: determine input Q-format
        self.modules.append({
            "type": LayerType.INPUT,
            "ignore": True,
            "params": {
                "in_dim": self.in_shape[1],
                "in_ch": self.in_shape[0],
                "out_dim": self.in_shape[1],
                "out_ch": self.in_shape[0],
                "out_ibits": self.databits,
                "out_fbits": 0
            }
        })

    def _quantize(self, weight, bias, in_format, out_format):
        # Quantize weights/biases
        weight, weight_ibits, weight_fbits = quantize(weight.flatten(), self.databits)
        bias, bias_ibits, bias_fbits = quantize(bias.flatten(), self.databits)

        print(f"    Input: Q{in_format[0]}.{in_format[1]}")
        print(f"    Weights: Q{weight_ibits}.{weight_fbits}")
        print(f"    Bias: Q{bias_ibits}.{bias_fbits}")
        print(f"    Optimal activation: Q{out_format[0]}.{out_format[1]}")

        # Compute Q-formats
        params = {}
        mul_fbits = weight_fbits + in_format[1] 
        params["out_rshift"] = max(mul_fbits - out_format[1], 0)
        params["in_ibits"] = in_format[0]
        params["in_fbits"] = in_format[1]
        params["out_fbits"] = mul_fbits - params["out_rshift"] 
        params["out_ibits"] = self.databits - params["out_fbits"]
        params["bias_lshift"] = max(bias_fbits - mul_fbits, 0)

        return [int(w) for w in weight], [int(b) for b in bias], params

    def convert_conv2d(self, mod):
        print(f"Converting 2D convolution layer ({mod}).")
        k_dim = mod.kernel_size[0]
        padding = mod.padding[0]
        stride = mod.stride[0]
        prev = self.modules[-1]
        prev_out_dim = prev["params"]["out_dim"]
        out_dim = int((prev_out_dim - k_dim + 2 * padding) / stride) + 1

        # Apply reordering and quantization on weights/bias
        weight, bias = reorder_conv2d(mod.weight, mod.bias) 
        weight, bias, q_params = self._quantize(weight, bias,
                                                (prev["params"]["out_ibits"],
                                                 prev["params"]["out_fbits"]),
                                                self.act_formats.pop(0))

        params = {
            "in_dim": prev_out_dim,
            "in_ch": mod.in_channels,
            "out_dim": out_dim, 
            "out_ch": mod.out_channels,
            "ker_dim": k_dim, 
            "stride": stride,
            "padding": padding,
        }
        params.update(q_params)

        weights = {
            "weight": weight,
            "bias": bias 
        }
    
        in_space = params["in_ch"] * params["in_dim"]**2
        out_space = params["out_ch"] * params["out_dim"]**2
        space = {
            "io": max(in_space, out_space),
            "tmp": 2 * params["in_ch"] * k_dim**2
        }

        prev_type = self.modules[-1]["type"]
        my_type = LayerType.CONV_RGB if prev_type == LayerType.INPUT else LayerType.CONV
        self.modules.append({
            "type": my_type,
            "mod": mod,
            "id": self._id_gen(my_type),
            "params": params,
            "weights": weights,
            "space": space
        })

    def convert_relu(self, mod):
        print(f"Converting ReLU layer ({mod})")
        prev_out_dim = self.modules[-1]["params"]["out_dim"]
        prev_out_ch = self.modules[-1]["params"]["out_ch"]
        self.modules.append({
            "type": LayerType.RELU,
            "id": self._id_gen(LayerType.RELU),
            "mod": mod,
            "params": {
                "in_dim": prev_out_dim,
                "in_ch": prev_out_ch,
                "out_dim": prev_out_dim,
                "out_ch": prev_out_ch,
                "out_ibits": self.modules[-1]["params"]["out_ibits"],
                "out_fbits": self.modules[-1]["params"]["out_fbits"]
            },
            "space": {
                "io": prev_out_ch * prev_out_dim**2,
                "tmp": 0
            }
        })

    def convert_maxpool2d(self, mod):
        print(f"Converting maxpool layer ({mod}).")
        k_dim = mod.kernel_size
        padding = mod.padding
        stride = mod.stride
        prev_out_dim = self.modules[-1]["params"]["out_dim"]
        prev_out_ch = self.modules[-1]["params"]["out_ch"]
        out_dim = (prev_out_dim - k_dim + 2 * padding) // stride + 1

        params = {
            "in_dim": prev_out_dim,
            "in_ch": prev_out_ch,
            "out_dim": out_dim,
            "out_ch": prev_out_ch,
            "ker_dim": k_dim,
            "stride": stride,
            "padding": max(padding - 1, 0),  # CMSIS-NN requires one less padding
            "out_ibits": self.modules[-1]["params"]["out_ibits"],
            "out_fbits": self.modules[-1]["params"]["out_fbits"]
        }

        in_space = params["in_ch"] * params["in_dim"]**2
        out_space = params["out_ch"] * params["out_dim"]**2
        space = {
            "io": max(in_space, out_space),
            "tmp": 0  # The CMSIS-NN maxpool implementation does not use tmp space
        }

        self.modules.append({
            "type": LayerType.MAXPOOL,
            "id": self._id_gen(LayerType.MAXPOOL),
            "mod": mod,
            "params": params,
            "space": space
        })

    def convert_flatten(self, mod):
        print(f"Converting flatten layer ({mod}).")
        prev_out_dim = self.modules[-1]["params"]["out_dim"]
        prev_out_ch = self.modules[-1]["params"]["out_ch"]

        params = {
            "in_dim": prev_out_dim,
            "in_ch": prev_out_ch,
            "out_dim": prev_out_ch * prev_out_dim**2,
            "out_ch": 1,
            "out_ibits": self.modules[-1]["params"]["out_ibits"],
            "out_fbits": self.modules[-1]["params"]["out_fbits"]
        }

        self.modules.append({
            "type": LayerType.FLATTEN,
            "id": self._id_gen(LayerType.FLATTEN),
            "mod": mod,
            "params": params
        })



    def convert_linear(self, mod):
        print(f"Converting linear layer ({mod}).")
        prev = self.modules[-1]
        prev_out_dim = prev["params"]["out_dim"]
        prev_out_ch = prev["params"]["out_ch"]
        assert prev_out_dim == mod.in_features
        assert prev_out_ch == 1

        # Apply reordering and quantization on weights/bias
        weight, bias = mod.weight, mod.bias
        weight, bias, q_params = self._quantize(weight, bias,
                                                (prev["params"]["out_ibits"],
                                                 prev["params"]["out_fbits"]),
                                                self.act_formats.pop(0))

        params = {
            "in_dim": mod.in_features,
            "in_ch": 1,
            "out_dim": mod.out_features,
            "out_ch": 1,
        }
        params.update(q_params)

        weights = {
            "weight": weight,
            "bias": bias 
        }

        common_space = max(params["in_dim"], params["out_dim"])
        space = {
            "io": common_space,
            "tmp": common_space
        }

        self.modules.append({
            "type": LayerType.FC,
            "id": self._id_gen(LayerType.FC),
            "mod": mod,
            "params": params,
            "weights": weights,
            "space": space
        })

    def fold_batchnorm(self, mod):
        print(f"Converting batchnorm layer ({mod}).")
        prev = self.modules[-1]
        bn_means, bn_vars = self.bn_mean_vars.pop(0)
        if type(mod) == nn.BatchNorm2d and \
           (prev["type"] == LayerType.CONV_RGB or prev["type"] == LayerType.CONV):
            weight, bias = reorder_conv2d(prev["mod"].weight, prev["mod"].bias)
        elif type(mod) == nn.BatchNorm1d and prev["type"] == LayerType.FC:
            weight, bias = prev["mod"].weight, prev["mod"].bias
        else:
            assert False, "Unable to fold {mod} into previous layer!"
        
        # weights = weights * (gamma / sqrt(var + eps))
        var_eps = bn_vars + mod.eps
        weight_factor = mod.weight * var_eps.rsqrt()
        for i, f in enumerate(weight_factor):
            weight[i] *= f.item()

        # bias = (gamma / sqrt(var + eps)) * (bias - mu) + beta 
        bias_centered = bias - bn_means
        bias = weight_factor * bias_centered + mod.bias

        if type(mod) == nn.BatchNorm2d:
            weight, bias = reorder_conv2d(weight, bias)
        
        weight, bias, q_params = self._quantize(weight, bias,
                                                (prev["params"]["in_ibits"],
                                                 prev["params"]["in_fbits"]),
                                                self.act_formats.pop(0))
        prev["params"].update(q_params)
        prev["weights"].update({
            "weight": weight,
            "bias": bias
        })

    def convert_softmax(self, mod):
        print(f"Converting softmax layer ({mod}).")
        prev = self.modules[-1]
        assert prev["params"]["out_ch"] == 1
        self.modules.append({
            "type": LayerType.SOFTMAX,
            "id": self._id_gen(LayerType.SOFTMAX),
            "params": {
                "in_dim": prev["params"]["out_dim"],
                "in_ch": 1,
                "out_dim": prev["params"]["out_dim"],
                "out_ch": 1,
                "out_ibits": prev["params"]["out_ibits"],
                "out_fbits": prev["params"]["out_fbits"]
            }
        })

    def convert(self, mod):
        if type(mod) == nn.Conv2d:
            self.convert_conv2d(mod)
        elif type(mod) == nn.ReLU:
            self.convert_relu(mod)
        elif type(mod) == nn.MaxPool2d:
            self.convert_maxpool2d(mod)
        elif type(mod) == nn.Flatten:
            self.convert_flatten(mod)
        elif type(mod) == nn.Linear:
            self.convert_linear(mod)
        elif type(mod) == nn.BatchNorm2d or type(mod) == nn.BatchNorm1d:
            self.fold_batchnorm(mod)
        elif type(mod) == nn.Softmax:
            self.convert_softmax(mod)
        elif type(mod) == nn.Sequential:
            print("Finished converting", mod)
        else:
            print("Unknown module:", mod)


def convert_parameters(model, datatype, act_hists, bn_mean_vars, quant_range):
    converter = Converter(static_in_shape, static_out_shape,
                          datatype, act_hists, bn_mean_vars, quant_range)
    model.apply(converter.convert)
    return converter.modules


############################################################################
# Code generation
############################################################################

def get_layer_name(mod):
    return f"{mod['type'].value}{mod['id']}".upper()


def get_layer_macro_name(mod, data_name):
    return f"{get_layer_name(mod)}_{data_name}".upper()


def get_layer_var_name(mod, data_name):
    return f"{get_layer_name(mod)}_{data_name}".lower()    


def write_macros(h, mod, macro_dict):
    h.code.append(C.comment(get_layer_name(mod)))
    for name, val in sorted(macro_dict.items(), key=lambda m: m[0]):
        full_name = get_layer_macro_name(mod, name) 
        if type(val) == list:
            val = "{" + ", ".join(str(v) for v in val) + "}"
        define = C.define(full_name, val)
        h.code.append(define)
    h.code.append(C.blank())

def write_header_files(mods, param_header, weight_header):
    ph = C.hfile(param_header)
    wh = C.hfile(weight_header)
    ph.code.append(C.blank())
    wh.code.append(C.blank())
    for mod in mods:
        if "ignore" in mod and mod["ignore"]:
            # FIXME: ignore some layers
            continue
        if "params" in mod:
            write_macros(ph, mod, mod["params"])
        if "weights" in mod:
            write_macros(wh, mod, mod["weights"])
    with open(param_header, "w") as f:
        f.write(str(ph))
    print("Wrote parameter file:", param_header)
    with open(weight_header, "w") as f:
        f.write(str(wh))
    print("Wrote weight file:", weight_header)


def gen_input(b, mod, buf_in, buf_out, buf_tmp, param_in, *args):
    # TODO: may need preprocessing here
    f = C.fcall("memcpy", [
        buf_in.name,
        param_in.name,
        str(mod["params"]["in_dim"]) + " * " +
        str(mod["params"]["in_dim"]) + " * " +
        str(mod["params"]["in_ch"]) + " * " +
        f"sizeof({param_in.typename})"
    ])
    b.append(C.statement(f))
    return buf_in, buf_out


def gen_common_conv_code(b, mod, name, buf_in, buf_out, buf_tmp):
    f = C.fcall(name, [
        buf_in.name,
        get_layer_macro_name(mod, "in_dim"),
        get_layer_macro_name(mod, "in_ch"),
        get_layer_var_name(mod, "weight"),
        get_layer_macro_name(mod, "out_ch"),
        get_layer_macro_name(mod, "ker_dim"),
        get_layer_macro_name(mod, "padding"),
        get_layer_macro_name(mod, "stride"),
        get_layer_var_name(mod, "bias"),
        get_layer_macro_name(mod, "bias_lshift"),
        get_layer_macro_name(mod, "out_rshift"),
        buf_out.name,
        get_layer_macro_name(mod, "out_dim"),
        f"(q15_t *) {buf_tmp.name}",
        "NULL"
    ])
    b.append(C.statement(f))
    return buf_out, buf_in


def gen_conv_rbg_code(b, mod, buf_in, buf_out, buf_tmp, *args):
    return gen_common_conv_code(b, mod, "arm_convolve_HWC_q7_RGB", buf_in, buf_out, buf_tmp)


def gen_conv_code(b, mod, buf_in, buf_out, buf_tmp, *args):
    return gen_common_conv_code(b, mod, "arm_convolve_HWC_q7_basic", buf_in, buf_out, buf_tmp)


def gen_relu_code(b, mod, buf_in, buf_out, buf_tmp, *args):
    f = C.fcall("arm_relu_q7", [
        buf_in.name,
        get_layer_macro_name(mod, "in_dim") + " * " +
        get_layer_macro_name(mod, "in_dim") + " * " +
        get_layer_macro_name(mod, "in_ch")
    ])
    b.append(C.statement(f))
    return buf_in, buf_out


def gen_maxpool_code(b, mod, buf_in, buf_out, buf_tmp, *args):
    f = C.fcall("arm_maxpool_q7_HWC", [
        buf_in.name,
        get_layer_macro_name(mod, "in_dim"),
        get_layer_macro_name(mod, "in_ch"),
        get_layer_macro_name(mod, "ker_dim"),
        get_layer_macro_name(mod, "padding"),
        get_layer_macro_name(mod, "stride"),
        get_layer_macro_name(mod, "out_dim"),
        buf_tmp.name,
        buf_out.name
    ])
    b.append(C.statement(f))
    return buf_out, buf_in


def gen_fc_code(b, mod, buf_in, buf_out, buf_tmp, *args):
    f = C.fcall("arm_fully_connected_q7", [
        buf_in.name,
        get_layer_var_name(mod, "weight"),
        get_layer_macro_name(mod, "in_dim"),
        get_layer_macro_name(mod, "out_dim"),
        get_layer_macro_name(mod, "bias_lshift"),
        get_layer_macro_name(mod, "out_rshift"),
        get_layer_var_name(mod, "bias"),
        buf_out.name,
        f"(q15_t *) {buf_tmp.name}"
    ])
    b.append(C.statement(f))
    return buf_out, buf_in


def gen_softmax_code(b, mod, buf_in, buf_out, buf_tmp, *args):
    f = C.fcall("arm_softmax_q7", [
        buf_in.name,
        get_layer_macro_name(mod, "in_dim"),
        buf_in.name
    ])
    b.append(C.statement(f))
    return buf_in, buf_out


def gen_forward_pass_code(b, mods, param_in, param_out):
    # Calculate size of working buffers
    # TODO: could optimize space usage better here instead of allocating double the max
    io_buf_half_len = 0
    tmp_buf_len = 0
    for mod in mods:
        if "space" in mod:
            io_buf_half_len = max(io_buf_half_len, mod["space"]["io"])
            tmp_buf_len = max(tmp_buf_len, mod["space"]["tmp"])

    # Define working buffers: one for input/output and one for temporary space
    io_buf = C.variable("io_buf", "q7_t", static=True, array=2 * io_buf_half_len)
    tmp_buf = C.variable("tmp_buf", "q7_t", static=True, array=tmp_buf_len)
    b.append(C.statement(str(io_buf)))
    b.append(C.statement(str(tmp_buf)))
    b.append(C.blank())

    # Define pointers into first and second half of input/output buffer
    io_buf_1 = C.variable("io_buf1", "q7_t", pointer=True)
    io_buf_2 = C.variable("io_buf2", "q7_t", pointer=True)
    b.append(C.statement(f"{io_buf_1} = &{io_buf.name}[0]"))
    b.append(C.statement(f"{io_buf_2} = &{io_buf.name}[{io_buf_half_len}]"))
    b.append(C.blank())

    # Iterate over layers, generating code for forward pass
    layers = {
        LayerType.INPUT: gen_input,
        LayerType.CONV_RGB: gen_conv_rbg_code,
        LayerType.CONV: gen_conv_code,
        LayerType.RELU: gen_relu_code,
        LayerType.FC: gen_fc_code,
        LayerType.SOFTMAX: gen_softmax_code,
    }
    buf_in = io_buf_1
    buf_out = io_buf_2
    for mod in mods:
        t = mod["type"]
        if t in layers:
            buf_in, buf_out = layers[t](b, mod, buf_in, buf_out,
                                        tmp_buf, param_in, param_out)
            b.append(C.blank())

    # Copy final data to output
    final_dim = get_layer_macro_name(mods[-1], "out_dim")
    cp = C.fcall("memcpy", [
        param_out.name,
        buf_in.name,
        f"{final_dim} * sizeof(q7_t)"
    ])
    b.append(C.statement(cp))
    b.append(C.blank())
    b.append(C.statement("return 0"))


def write_code(mods, model_source, model_header, param_header, weight_header):
    f = C.cfile(model_source)
    f.code.append(C.comment("Autogenerated model source code."))
    f.code.append(C.blank())

    # Write #includes
    f.code.append(C.include("arm_nnfunctions.h"))
    f.code.append(C.include(model_header.name))
    f.code.append(C.include(param_header.name))
    f.code.append(C.include(weight_header.name))
    f.code.append(C.blank())

    # Write static variables
    for mod in mods:
        if "weights" in mod:
            for name in sorted(mod["weights"].keys()):
                var_name = get_layer_var_name(mod, name)
                macro_name = get_layer_macro_name(mod, name)
                variable = C.variable(var_name, "q7_t", static=True, array="")
                f.code.append(C.statement(f"{variable} = {macro_name}"))
            f.code.append(C.blank())

    # Write forward pass function
    fw = C.function("nn_forward_pass", typename="int")
    param_in = C.variable("img", "uint8_t", pointer=True)
    param_out = C.variable("out", "q7_t", pointer=True)
    fw.add_param(param_in)
    fw.add_param(param_out)
    f.code.append(fw)

    # Write forward pass function body
    fw_body = C.block(innerIndent=4)
    gen_forward_pass_code(fw_body, mods, param_in, param_out)
    f.code.append(fw_body)

    with open(model_source, "w") as sf:
        sf.write(str(f))
    print("Wrote model source code:", model_source)

    h = C.hfile(model_header)
    h.code.append(C.blank())
    h.code.append(C.sysinclude("stdint.h"))
    h.code.append(C.include("arm_math.h"))
    h.code.append(C.blank())
    h.code.append(C.statement(fw))

    with open(model_header, "w") as f:
        f.write(str(h))
    print("Wrote model header:", model_header)


def parse_arguments():
    p = argparse.ArgumentParser(description="Convert a saved FastAI/PyTorch model to CMSIS-NN")
    p.add_argument("modelpath")
    p.add_argument("actpath")
    p.add_argument("outpath")
    p.add_argument("--name", default=None)
    size_group = p.add_mutually_exclusive_group(required=True)
    size_group.add_argument("--q7", action="store_true")
    size_group.add_argument("--q15", action="store_true")
    p.add_argument("--qwrange", type=float, default=1.0)
    return p.parse_args()


def make_valid_identifier(name):
    n = re.sub(r"[^_A-Za-z0-9]", "_", name)
    if n[0] in string.digits:
        n = "_" + n
    return n


def main():
    # Parse and prepare arguments
    args = parse_arguments()
    model_path = Path(args.modelpath)
    if args.name is None:
        model_name = Path(model_path.name).stem
    else:
        model_name = args.name
    model_name = make_valid_identifier(model_name)
    act_path = Path(args.actpath)
    out_path = Path(args.outpath)
    param_header = out_path/f"{model_name}_params.h"
    weight_header = out_path/f"{model_name}_weights.h"
    model_header = out_path/f"{model_name}_model.h"
    model_source = out_path/f"{model_name}_model.c"
    qt = "q7_t" if args.q7 else "q15_t"
    qwrange = args.qwrange

    # Convert model
    print(f"Loading model {model_path}.")
    model = shared_model.create_model()
    load_model(model_path, model, None, with_opt=False) 
    print(f"Loading activation histogram {act_path}")
    act_hist, bn_mean_vars = load_run_stats(act_path)
    print("Converting model parameters.")
    modules = convert_parameters(model, qt, act_hist, bn_mean_vars, qwrange)

    # Generate source code/headers
    write_header_files(modules, param_header, weight_header)
    write_code(modules, model_source, model_header, param_header, weight_header)


if __name__ == "__main__":
    main()
