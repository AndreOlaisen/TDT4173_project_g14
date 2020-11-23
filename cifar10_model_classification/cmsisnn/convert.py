import os
import re
import sys
import string
import math
import argparse
import json
import typing
import copy
import itertools
import torch
import torch.nn as nn
import torchvision.transforms
from pathlib import Path
from collections import Counter, namedtuple
from enum import Enum
from . import cfile as C
from model_export import make_model_filename, make_stats_filename, make_transform_filename


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


QFormat = namedtuple("qformat", ["ibits", "fbits"])
Dimension = namedtuple("dimension", ["dim", "ch"])
Space = namedtuple("space", ["i", "o", "tmp"])
Weight = namedtuple("weight", ["weight", "bias"])
ShiftParams = namedtuple("shift", ["out_rshift", "bias_lshift"])
KernelParams = namedtuple("kernel", ["dim", "stride", "padding"])
NormalizeParams = namedtuple("normalize", ["mean", "shift"])


class Layer:
    def __init__(self, layer_type, identifier, module, prev=None):
        self.type = layer_type
        self.identifier = identifier
        self.module = module
        self.prev = prev
        self.parameters = {}
        self.inherit = {}

    @staticmethod
    def _param_name(param, cat):
        if type(param) == type:
            param_name = param.__name__
        else:
            param_name = param.__class__.__name__
        if cat is not None:
            return f"{cat}_{param_name}"
        else:
            return param_name

    def get_layer_name(self):
        return f"{self.type.name}{self.identifier}"

    def _full_member_name(self, member, param, cat):
        return f"{self.get_layer_name()}_{self._param_name(param, cat)}_{member}"

    def has_param(self, param, cat=None):
        param_name = self._param_name(param, cat)
        return param_name in self.parameters or param_name in self.inherit

    def get_param_var_name(self, member, param, cat=None):
        return self._full_member_name(member, param, cat).lower()

    def get_param_macro_name(self, member, param, cat=None):
        return self._full_member_name(member, param, cat).upper()

    def get_param(self, param, cat=None):
        param_name = self._param_name(param, cat)
        p = self.parameters.get(param_name, None)
        if p is not None:
            return p
        if self.prev is not None and param_name in self.inherit:
            prev_name, prev_cat = self.inherit[param_name]
            return self.prev.get_param(prev_name, prev_cat)
        raise KeyError(f"Could not find parameter {param_name}!")

    def set_param(self, param, cat=None):
        param_name = self._param_name(param, cat)
        self.parameters[param_name] = param

    def set_params(self, *params, cat=None):
        for p in params:
            self.set_param(p, cat)

    def inherit_param(self, prev_param, prev_cat, param, cat):
        param_name = self._param_name(param, cat)
        self.inherit[param_name] = (prev_param, prev_cat) 

    def to_debug_obj(self):
        inherited = dict((k, self.prev.get_param(it[0], it[1])) for k, it in self.inherit.items())
        full_parameters = {}
        full_parameters.update(self.parameters)
        full_parameters.update(inherited)
        return {
            "name": self.get_layer_name(),
            "parameters": full_parameters,
        }


def dump_layer_json(file_name, layers):
    def json_default(obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        else:
            return obj._asdict()
    objects = [l.to_debug_obj() for l in layers]
    with open(file_name, "w") as f:
        json.dump(objects, f, indent=4, default=json_default)


def reorder_conv2d(weight, bias):
    """
        Reorder weight format from:
        PyTorch [out_channels, in_channels, kernel_height, kernel_width]
        (CHW format)
        to
        CMSIS-NN [out_channels, kernel_height, kernel_width, in_channels]
        (HWC format)
    """
    return weight.permute(0, 2, 3, 1), bias


def reorder_linear(weight, bias, prev_dim):
    """
        Reorder weights from:
        PyTorch [out_channels, in_channels * input_height * input_width]
        (CHW format)
        to
        CMSIS-NN [out_channels, input_height * input_width * in_channels] 
        (HWC format)
    """
    if prev_dim.ch > 1:
        print("Input dimension:", prev_dim)
        print("Weights before:")
        print(weight)
        weight_shape = weight.shape
        input_shape = (weight_shape[0], prev_dim.ch, prev_dim.dim, prev_dim.dim)
        w = weight.reshape(input_shape)
        w = w.permute(0, 2, 3, 1)
        print("Permuted weight shape:", w.shape)
        w = w.reshape(weight_shape)
        print("Weights after:")
        print(w)
        return w, bias
    else:
        return weight, bias


def calc_qformat(weights, databits):
    min_wt_abs = abs(weights.min().item())
    max_wt_abs = abs(weights.max().item())
    ibits = max(int(math.ceil(math.log2(max(min_wt_abs, max_wt_abs)))), 0)
    fbits = databits - ibits
    return QFormat(ibits, fbits)


def calc_quantized(weights, fbits, truncate=False):
    if isinstance(weights, torch.Tensor):
        w = weights * (2**fbits)
        if not truncate:
            return w.round()
        else:
            return w.floor()
    else:
        if not truncate:
            f = round 
        else:
            f = math.floor
        return [f(w * (2**fbits)) for w in weights]


def quantize(weights, databits):
    qformat = calc_qformat(weights, databits)
    quant_weight = calc_quantized(weights, qformat.fbits) 
    return quant_weight, qformat


def convert_transforms(transforms, input_shape, databits):
    if len(input_shape) != 3:
        raise RuntimeError("Only inputs of ch x dim x dim are supported.")
    if isinstance(transforms, torchvision.transforms.Compose):
        transforms = transforms.transforms
    dim_in = Dimension(dim=input_shape[1], ch=input_shape[0])
    dim_out = Dimension(dim=input_shape[1], ch=input_shape[0])
    q_in = QFormat(8, 0)
    q_out = QFormat(databits, 0)
    value_range = torch.tensor([[0.0, 255.0] for _ in range(dim_in.ch)])
    input_layer = Layer(LayerType.INPUT, 0, transforms, None)
    for t in transforms:
        if type(t) == torchvision.transforms.ToTensor:
            # Should convert input from [0, 255] to [0, 1]
            q_in = QFormat(0, 8)
            value_range /= 255.0
            q_out = calc_qformat(value_range, databits)
            pass
        elif type(t) == torchvision.transforms.Normalize:
            # Should perform out = (input - mean) / std for each channel
            mean = torch.tensor([[m] for m in t.mean])
            std = torch.tensor([[s] for s in t.std])
            value_range = (value_range - mean) / std
            q_out = calc_qformat(value_range, databits)
            mean_quant = calc_quantized(t.mean, q_in.fbits)
            shift = q_in.fbits - q_out.fbits
            assert shift >= 0, "Bad shift value"
            input_layer.set_param(NormalizeParams(mean_quant, shift))
        else:
            raise RuntimeError(f"Unable to convert transform {t}!")
    input_layer.set_params(dim_in, q_in, cat="in")
    input_layer.set_params(dim_out, q_out, cat="out")
    return input_layer


class Converter:
    def __init__(self, transforms, run_stats, datatype, quant_range=1.0, use_opt=False):
        self.layers = []
        self.id_counter = Counter()
        self.run_stats = run_stats
        if datatype == "q7_t":
            self.databits = 8 - 1  # 8 bits minus sign bit
        else:
            self.databits = 16 - 1  # 16 bits minus sign bit
        self.batchnorm_mvs = [mv for mv in run_stats.batchnorm_mv]
        self.act_formats = self._calc_act_formats(run_stats.act_hist, quant_range)
        input_layer = convert_transforms(transforms, run_stats.input_shape, self.databits)
        self.layers.append(input_layer)

    def _id_gen(self, layer_type):
        """ Generate unique ID number for layer with type layer_type """ 
        count = self.id_counter[layer_type]
        self.id_counter[layer_type] += 1
        return count

    def _calc_act_formats(self, act_hists, pct_threshold=1.0):
        formats = []
        for hist in act_hists:
            cumulative = itertools.accumulate(hist)
            hist_sum = sum(hist)

            # FIXME
            cumulative = [c / hist_sum for c in cumulative]
            for i, (val, pct) in enumerate(zip(hist[1:self.databits + 1], cumulative[:self.databits + 2])):
                if val == 0.0 or pct >= pct_threshold:
                    if val != 0.0:
                        print(f"Clipping output at {pct * 100.0}%.")
                    formats.append(QFormat(i, self.databits - i))
                    break
            else:
                print(f"Warning: activation values exceed range ({self.databits} bits)!")
                print(f"Lost apx. {100.0 - cumulative[self.databits + 1]}% of activations.")
                formats.append(QFormat(self.databits, 0))

        return formats

    def _new_default_layer(self, layer_type, module, prev=None):
        layer = Layer(layer_type, self._id_gen(layer_type), module, prev)
        layer.inherit_param(Dimension, "out", Dimension, "in")
        layer.inherit_param(QFormat, "out", QFormat, "in")
        layer.set_param(layer.get_param(Dimension, "in"), "out")
        layer.set_param(layer.get_param(QFormat, "in"), "out")
        return layer

    def _quantize(self, weight, bias, qformat_in, qformat_out):
        """
        Quantize weights/biases
        """
        # Quantize weights/biases
        weight, qformat_weight = quantize(weight, self.databits)
        bias, qformat_bias = quantize(bias, self.databits)

        print(f"    QFormats:")
        print(f"    Input: {qformat_in}")
        print(f"    Weights: {qformat_weight}")
        print(f"    Bias: {qformat_bias}")
        print(f"    Optimal activation: {qformat_out}")

        # Compute Q-formats
        mul_fbits = qformat_weight.fbits + qformat_in.fbits
        out_rshift = mul_fbits - qformat_out.fbits
        if out_rshift < 0:
            print("Warning: unable to rshift convolution result of",
                  f"Qx.{mul_fbits} to Q{qformat_out.ibits}.{qformat_out.fbits}.")
            out_rshift = 0
        out_fbits = mul_fbits - out_rshift
        out_ibits = self.databits - out_fbits
        bias_lshift = mul_fbits - qformat_bias.fbits
        if bias_lshift < 0:
            print("Warning: unable to shift bias of",
                  f"Q{qformat_bias.ibits}.{qformat_bias.ibits} to Qx.{mul_fbits}.")
            bias_lshift = 0
        qformat_out = QFormat(out_ibits, out_fbits)
        shift_params = ShiftParams(out_rshift, bias_lshift)
        q_weight = Weight(weight, bias)
        return q_weight, qformat_out, shift_params

    @staticmethod
    def _calc_kernel_out_dim(in_dim, k_params):
        return int((in_dim.dim - k_params.dim + 2 * k_params.padding) / k_params.stride) + 1

    @staticmethod
    def _calc_2d_space(in_dim, out_dim):
        return in_dim.ch * in_dim.dim**2, out_dim.ch * out_dim.dim**2

    def convert_conv2d(self, mod, prev):
        print(f"Converting 2D convolution layer ({mod}).")
        # The first conv layer should be conv_rgb, others should be conv.
        conv_type = LayerType.CONV_RGB if prev.type == LayerType.INPUT else LayerType.CONV
        layer = self._new_default_layer(conv_type, mod, prev)
        # NOTE: Assumes square parameters
        k_params = KernelParams(mod.kernel_size[0], mod.stride[0], mod.padding[0])
        layer.set_param(k_params)
        # Calculate output dimensions
        in_dim = layer.get_param(Dimension, "in")
        out_dim = Dimension(self._calc_kernel_out_dim(in_dim, k_params), mod.out_channels)
        layer.set_param(out_dim, "out") 
        # Apply quantization on weights/bias (reordering is performed at code generation time)
        act_format = self.act_formats.pop(0)
        qformat_in = layer.get_param(QFormat, "in")
        weight, bias = reorder_conv2d(mod.weight.data, mod.bias.data)
        q_weight, qformat_out, shift_params = self._quantize(weight, bias,
                                                             qformat_in, act_format)
        layer.set_param(q_weight)
        layer.set_param(qformat_out, "out")
        layer.set_param(shift_params)
        # Calculate space requirements
        in_space, out_space = self._calc_2d_space(in_dim, out_dim)
        tmp_space = 2 * in_dim.ch * k_params.dim**2  # Buffer space required by CMSIS-NN conv
        layer.set_param(Space(in_space, out_space, tmp_space))
        self.layers.append(layer)

    def convert_relu(self, mod, prev):
        print(f"Converting ReLU layer ({mod})")
        layer = self._new_default_layer(LayerType.RELU, mod, prev)
        in_dim = layer.get_param(Dimension, "in")
        io_space = in_dim.ch * in_dim.dim**2
        layer.set_param(Space(io_space, io_space, 0))
        self.layers.append(layer)

    def convert_maxpool2d(self, mod, prev):
        print(f"Converting maxpool layer ({mod}).")
        layer = self._new_default_layer(LayerType.MAXPOOL, mod, prev)
        # CMSIS-NN requires one less padding for pooling operations (for some reason)
        k_params = KernelParams(mod.kernel_size, mod.stride, max(mod.padding - 1, 0))
        layer.set_param(k_params)
        in_dim = layer.get_param(Dimension, "in")
        # FIXME: better solution for weird padding
        k_params_c = KernelParams(k_params.dim, k_params.stride, k_params.padding + 1)
        out_dim = Dimension(self._calc_kernel_out_dim(in_dim, k_params_c), in_dim.ch)
        layer.set_param(out_dim, "out")
        # Calculate space requirements
        in_space, out_space = self._calc_2d_space(in_dim, out_dim)
        layer.set_param(Space(in_space, out_space, 0))
        self.layers.append(layer)

    def convert_flatten(self, mod, prev):
        print(f"Ignoring flatten layer ({mod}).")

    def convert_linear(self, mod, prev):
        print(f"Converting linear layer ({mod}).")
        layer = self._new_default_layer(LayerType.FC, mod, prev)
        in_dim = layer.get_param(Dimension, "in")
        if in_dim.ch > 1 and in_dim.ch * in_dim.dim**2 != mod.in_features:
            raise RuntimeError(f"Bad input dimensions {in_dim.ch}, {in_dim.dim} to linear layer!")

        print("Module biases:", mod.bias)
        qformat_in = layer.get_param(QFormat, "in")
        act_format = self.act_formats.pop(0)
        weight, bias = reorder_linear(mod.weight.data, mod.bias.data, in_dim)
        q_weight, qformat_out, shift_params = self._quantize(weight, bias,
                                                             qformat_in, act_format)
        out_dim = Dimension(mod.out_features, 1)
        layer.set_param(q_weight)
        layer.set_params(out_dim, qformat_out, cat="out")
        layer.set_param(shift_params)

        common_space = max(mod.in_features, mod.out_features)
        layer.set_param(Space(mod.in_features, mod.out_features, common_space))
        self.layers.append(layer)

    def fold_batchnorm(self, mod, prev):
        print(f"Converting batchnorm layer ({mod}).")
        bn_means, bn_vars = self.batchnorm_mvs.pop(0)
        valid_2d = type(mod) == nn.BatchNorm2d and \
                   (prev.type == LayerType.CONV_RGB or prev.type == LayerType.CONV)
        valid_1d = type(mod) == nn.BatchNorm1d and prev.type == LayerType.FC
        if valid_1d or valid_2d:
            prev_weight = torch.tensor(prev.module.weight.data)
            prev_bias = torch.tensor(prev.module.bias.data)
        else:
            raise RuntimeError(f"Unable to fold {mod} into previous layer!")
        
        # weights = weights * (gamma / sqrt(var + eps))
        var_eps = bn_vars + mod.eps
        weight_factor = mod.weight.data * var_eps.rsqrt()
        for i, f in enumerate(weight_factor):
            prev_weight[i] *= f.item()

        # bias = (gamma / sqrt(var + eps)) * (bias - mu) + beta 
        prev_bias -= bn_means
        prev_bias *= weight_factor
        prev_bias += mod.bias

        if type(mod) == nn.BatchNorm2d:
            weight, bias = reorder_conv2d(prev_weight, prev_bias)
        else:
            prev_dim_in = prev.get_param(Dimension, "in")
            prev_dim_out = prev.get_param(Dimension, "out")
            weight, bias = reorder_linear(prev_weight, prev_bias, prev_dim_in)

        prev_in_format = prev.get_param(QFormat, "in")
        act_format = self.act_formats.pop(0)
        q_weight, qformat_out, shift_params = self._quantize(weight, bias, prev_in_format, act_format)
        prev.set_param(q_weight)
        prev.set_param(qformat_out, "out")
        prev.set_param(shift_params)

    def convert_softmax(self, mod, prev):
        print(f"Converting softmax layer ({mod}).")
        layer = self._new_default_layer(LayerType.SOFTMAX, mod, prev)
        in_dim = layer.get_param(Dimension, "in")
        if in_dim.ch != 1:
            raise RuntimeError(f"Invalid channels for softmax: {in_dim.ch}")
        layer.set_param(Space(in_dim.dim, in_dim.dim, 0))
        self.layers.append(layer)

    def convert(self, mod):
        prev = None if len(self.layers) == 0 else self.layers[-1]
        if type(mod) == nn.Conv2d:
            self.convert_conv2d(mod, prev)
        elif type(mod) == nn.ReLU:
            self.convert_relu(mod, prev)
        elif type(mod) == nn.MaxPool2d:
            self.convert_maxpool2d(mod, prev)
        elif type(mod) == nn.Flatten:
            self.convert_flatten(mod, prev)
        elif type(mod) == nn.Linear:
            self.convert_linear(mod, prev)
        elif type(mod) == nn.BatchNorm2d or type(mod) == nn.BatchNorm1d:
            self.fold_batchnorm(mod, prev)
        elif type(mod) == nn.Softmax:
            self.convert_softmax(mod, prev)
        elif type(mod) == nn.Sequential:
            print("Finished converting", mod)
        else:
            print("Unknown module:", mod)


def convert_parameters(model, transforms, run_stats, datatype, quant_range=1.0):
    converter = Converter(transforms, run_stats, datatype, quant_range)
    model.apply(converter.convert)
    return converter.layers


############################################################################
# Code generation
############################################################################

GenOutput = namedtuple("GenOutput", [
    "params",
    "weights",
    "statvars",
    "statvars_debug",
    "function"])

BufferParameters = namedtuple("BufferParameters", [
    "buf_in",
    "buf_out",
    "buf_tmp",
    "param_in",
    "param_out",
    "status"])


def get_datatype_prefix(datatype):
    return datatype[:-len("_t")]


def initializer_list(iterable, key):
    return "{" + ", ".join(str(key(v)) for v in iterable) + "}"


def gen_output_dump(layer, outputs, buffer):
    name = layer.get_layer_name()
    name_var = C.variable(f"{name.lower()}_name", "char",
                          static=True, const=True, array="")
    # Define name of layer
    outputs.statvars_debug.append(C.statement(f"{name_var} = \"{name}\""))
    # Add function call to dump layer activations
    space = layer.get_param(Space)
    fcall = C.fcall("nn_dump_activations", [
        name_var.name,
        buffer.name,
        space.o
    ])
    outputs.function.append(C.statement(fcall))


def gen_status_check(output, buffer):
    output.append(C.statement(C.fcall(
        "ARM_STATUS_CHECK", [buffer.name]
    )))


def gen_macros(layer, output, param, cat=None):
    values = layer.get_param(param, cat)
    for member in values._fields:
        name = layer.get_param_macro_name(member, param, cat)
        value = getattr(values, member)
        if isinstance(value, torch.Tensor):
            value = initializer_list(value.flatten(), lambda v: int(v.item()))
        if type(value) == list:
            value = initializer_list(value, lambda v: int(v))
        output.append(C.define(name, value))


def gen_layer_macros(layer, outputs):
    if layer.has_param(NormalizeParams):
        gen_macros(layer, outputs.params, NormalizeParams)
    gen_macros(layer, outputs.params, Dimension, "in")
    gen_macros(layer, outputs.params, QFormat, "in")
    gen_macros(layer, outputs.params, Dimension, "out")
    gen_macros(layer, outputs.params, QFormat, "out")
    if layer.has_param(ShiftParams):
        gen_macros(layer, outputs.params, ShiftParams)
    if layer.has_param(Weight):
        gen_macros(layer, outputs.weights, Weight)
    if layer.has_param(KernelParams):
        gen_macros(layer, outputs.weights, KernelParams)
    outputs.params.append(C.blank())


def gen_vars(layer, output, datatype, param, cat=None):
    variables = []
    values = layer.get_param(param, cat)
    for member in values._fields:
        var_name = layer.get_param_var_name(member, param, cat)
        macro_name = layer.get_param_macro_name(member, param, cat)
        var = C.variable(var_name, datatype, static=True, array="")
        statement = C.statement(f"{var} = {macro_name}")
        output.append(statement)
        variables.append(var)
    return variables


def gen_weight_bias_vars(layer, outputs, datatype):
    comment = C.comment(f" {layer.get_layer_name()} ")
    outputs.statvars.append(comment)
    return gen_vars(layer, outputs.statvars, datatype, Weight)


def gen_input_code(layer, outputs, buffers, datatype, debug=False):
    # TODO: Fix the implementation
    gen_layer_macros(layer, outputs)
    in_dim = layer.get_param(Dimension, "in")
    in_fbits_macro = layer.get_param_macro_name("fbits", QFormat, "in")
    out_fbits_macro = layer.get_param_macro_name("fbits", QFormat, "out")
    in_dim_macro = layer.get_param_macro_name("dim", Dimension, "in")
    in_ch_macro = layer.get_param_macro_name("ch", Dimension, "in")
    statements = [f"{buffers.param_in.name}[i + {i}]" for i in range(in_dim.ch)]
    ssat_bits = 8 if datatype == "q7_t" else 16
    if layer.has_param(NormalizeParams):
        mean_var_name = layer.get_param_var_name("mean", NormalizeParams)
        mean_macro = layer.get_param_macro_name("mean", NormalizeParams)
        mean = C.variable(mean_var_name, "int", static=True, const=True, array="")
        outputs.statvars.append(C.statement(f"{mean} = {mean_macro}"))
        shift_var_name = layer.get_param_var_name("shift", NormalizeParams)
        shift_macro = layer.get_param_macro_name("shift", NormalizeParams)
        shift = C.variable(shift_var_name, "int", static=True, const=True)
        outputs.statvars.append(C.statement(f"{shift} = {shift_macro}"))
        statements = [
            str(C.fcall("preprocess", [s, f"{mean.name}[{i}]", shift.name]))
            for i, s in enumerate(statements)
        ]
    statements = [
        C.statement(f"{buffers.buf_in.name}[i + {i}] = {s}")
        for i, s in enumerate(statements)
    ]
    loop = C.line("for (int i = 0; " +
                  f"i < {in_dim_macro} * {in_dim_macro} * {in_ch_macro}; " +
                  f"i += {in_ch_macro})")
    outputs.function.append(loop)
    block = C.block(innerIndent=4)
    block.extend(statements)
    outputs.function.append(block)
    layer.set_param(Space(0, in_dim.ch * in_dim.dim**2, 0))
    if debug:
        gen_output_dump(layer, outputs, buffers.buf_in)
    
    return buffers.buf_in, buffers.buf_out


def gen_common_conv_code(layer, outputs, buffers, datatype, name, debug=False):
    gen_layer_macros(layer, outputs)
    weight_var, bias_var = gen_weight_bias_vars(layer, outputs, datatype)
    f = C.fcall(name, [
        buffers.buf_in.name,
        layer.get_param_macro_name("dim", Dimension, "in"),
        layer.get_param_macro_name("ch", Dimension, "in"),
        weight_var.name,
        layer.get_param_macro_name("ch", Dimension, "out"),
        layer.get_param_macro_name("dim", KernelParams),
        layer.get_param_macro_name("padding", KernelParams),
        layer.get_param_macro_name("stride", KernelParams),
        bias_var.name,
        layer.get_param_macro_name("bias_lshift", ShiftParams),
        layer.get_param_macro_name("out_rshift", ShiftParams),
        buffers.buf_out.name,
        layer.get_param_macro_name("dim", Dimension, "out"),
        f"(q15_t *) {buffers.buf_tmp.name}",
        "NULL"
    ])
    outputs.function.append(C.statement(f"{buffers.status.name} = {f}"))
    gen_status_check(outputs.function, buffers.status)
    if debug:
        gen_output_dump(layer, outputs, buffers.buf_out)
    return buffers.buf_out, buffers.buf_in


def gen_conv_rbg_code(layer, outputs, buffers, datatype, debug=False):
    name = f"arm_convolve_HWC_{get_datatype_prefix(datatype)}_RGB"
    return gen_common_conv_code(layer, outputs, buffers, datatype, name, debug)


def gen_conv_code(layer, outputs, buffers, datatype, debug=False):
    name = f"arm_convolve_HWC_{get_datatype_prefix(datatype)}_basic"
    return gen_common_conv_code(layer, outputs, buffers, datatype, name, debug)


def gen_relu_code(layer, outputs, buffers, datatype, debug=False):
    gen_layer_macros(layer, outputs)
    f = C.fcall(f"arm_relu_{get_datatype_prefix(datatype)}", [
        buffers.buf_in.name,
        layer.get_param_macro_name("dim", Dimension, "in") + " * " +
        layer.get_param_macro_name("dim", Dimension, "in") + " * " +
        layer.get_param_macro_name("ch", Dimension, "in")
    ])
    outputs.function.append(C.statement(f))
    if debug:
        gen_output_dump(layer, outputs, buffers.buf_in)
    return buffers.buf_in, buffers.buf_out


def gen_maxpool_code(layer, outputs, buffers, datatype, debug=False):
    gen_layer_macros(layer, outputs)
    f = C.fcall(f"arm_maxpool_{get_datatype_prefix(datatype)}_HWC", [
        buffers.buf_in.name,
        layer.get_param_macro_name("dim", Dimension, "in"),
        layer.get_param_macro_name("ch", Dimension, "in"),
        layer.get_param_macro_name("dim", KernelParams),
        layer.get_param_macro_name("padding", KernelParams),
        layer.get_param_macro_name("stride", KernelParams),
        layer.get_param_macro_name("dim", Dimension, "out"),
        buffers.buf_tmp.name,
        buffers.buf_out.name
    ])
    outputs.function.append(C.statement(f))
    if debug:
        gen_output_dump(layer, outputs, buffers.buf_out)
    return buffers.buf_out, buffers.buf_in


def gen_fc_code(layer, outputs, buffers, datatype, debug=False):
    gen_layer_macros(layer, outputs)
    weight_var, bias_var = gen_weight_bias_vars(layer, outputs, datatype)
    dim_in_macro = layer.get_param_macro_name("dim", Dimension, "in")
    ch_in_macro = layer.get_param_macro_name("ch", Dimension, "in")
    f = C.fcall(f"arm_fully_connected_{get_datatype_prefix(datatype)}", [
        buffers.buf_in.name,
        weight_var.name,
        f"{dim_in_macro} * {dim_in_macro} * {ch_in_macro}",
        layer.get_param_macro_name("dim", Dimension, "out"),
        layer.get_param_macro_name("bias_lshift", ShiftParams),
        layer.get_param_macro_name("out_rshift", ShiftParams),
        bias_var.name,
        buffers.buf_out.name,
        f"(q15_t *) {buffers.buf_tmp.name}"
    ])
    outputs.function.append(C.statement(f"{buffers.status.name} = {f}"))
    gen_status_check(outputs.function, buffers.status)
    if debug:
        gen_output_dump(layer, outputs, buffers.buf_out)
    return buffers.buf_out, buffers.buf_in


def gen_softmax_code(layer, outputs, buffers, datatype, debug=False):
    gen_layer_macros(layer, outputs)
    f = C.fcall(f"arm_softmax_{get_datatype_prefix(datatype)}", [
        buffers.buf_in.name,
        layer.get_param_macro_name("dim", Dimension, "in"),
        buffers.buf_in.name
    ])
    outputs.function.append(C.statement(f))
    if debug:
        gen_output_dump(layer, outputs, buffers.buf_in)
    return buffers.buf_in, buffers.buf_out


def gen_layer_code(layers, outputs, param_in, param_out, datatype, debug=False):
    # Calculate size of working buffers
    # TODO: could optimize space usage better here instead of allocating double the max
    io_buf_half_len = 0
    tmp_buf_len = 0
    for l in layers:
        if l.has_param(Space):
            space = l.get_param(Space)
            io_buf_half_len = max(io_buf_half_len, space.i, space.o)
            tmp_buf_len = max(tmp_buf_len, space.tmp)

    # Define working buffers: one for input/output and one for temporary space
    io_buf = C.variable("io_buf", datatype, static=True, array=2 * io_buf_half_len)
    tmp_buf = C.variable("tmp_buf", datatype, static=True, array=tmp_buf_len)
    outputs.function.append(C.statement(str(io_buf)))
    outputs.function.append(C.statement(str(tmp_buf)))
    outputs.function.append(C.blank())

    # Define pointers into first and second half of input/output buffer
    io_buf_1 = C.variable("io_buf1", datatype, pointer=True)
    io_buf_2 = C.variable("io_buf2", datatype, pointer=True)
    outputs.function.append(C.statement(f"{io_buf_1} = &{io_buf.name}[0]"))
    outputs.function.append(C.statement(f"{io_buf_2} = &{io_buf.name}[{io_buf_half_len}]"))
    outputs.function.append(C.blank())

    status = C.variable("status", "arm_status")
    outputs.function.append(C.statement(str(status)))

    # Initialize debug dump
    if debug:
        outputs.function.append(C.statement(C.fcall("nn_dump_open")))

    layer_gen = {
        LayerType.INPUT: gen_input_code,
        LayerType.CONV_RGB: gen_conv_rbg_code,
        LayerType.CONV: gen_conv_code,
        LayerType.RELU: gen_relu_code,
        LayerType.MAXPOOL: gen_maxpool_code,
        LayerType.FC: gen_fc_code,
        LayerType.SOFTMAX: gen_softmax_code,
    }
    buffers = BufferParameters(io_buf_1, io_buf_2, tmp_buf, param_in, param_out, status)
    for l in layers:
        t = l.type
        if t in layer_gen:
            buf_in, buf_out = layer_gen[t](l, outputs, buffers, datatype, debug)
            buffers = BufferParameters(buf_in, buf_out, buffers.buf_tmp,
                                       buffers.param_in, buffers.param_out, buffers.status)
        else:
            print(f"Ignoring layer with type {t}.")

    # Copy final data to output
    final_dim = layers[-1].get_param_macro_name("dim", Dimension, "out")
    cp = C.fcall("memcpy", [
        param_out.name,
        buffers.buf_in.name,
        f"{final_dim} * sizeof({datatype})"
    ])
    outputs.function.append(C.statement(cp))
    outputs.function.append(C.blank())

    if debug:
        outputs.function.append(C.statement(C.fcall("nn_dump_close")))
        outputs.function.append(C.blank())

    outputs.function.append(C.statement("return 0"))


def write_code(layers, model_source, model_header, param_header,
               weight_header, datatype, debug=False):
    # Create static variable definition section
    outputs = GenOutput(params=C.sequence(), weights=C.sequence(),
                        statvars=C.sequence(), statvars_debug=C.sequence(),
                        function=C.sequence())

    # Create "nn_forward_pass" function parameters
    param_in = C.variable("img", "uint8_t", pointer=True)
    param_out = C.variable("out", datatype, pointer=True)

    # Fill the outputs based on the layers
    gen_layer_code(layers, outputs, param_in, param_out, datatype, debug)

    # Create parameter header file
    ph = C.hfile(param_header)
    ph.code.append(C.blank())
    ph.code.extend(outputs.params)

    # Create weight header file
    wh = C.hfile(weight_header)
    wh.code.append(C.blank())
    wh.code.extend(outputs.weights)

    ms = C.cfile(model_source)
    ms.code.append(C.comment("Autogenerated model source code."))
    ms.code.append(C.blank())

    # Write #includes
    ms.code.append(C.include("arm_nnfunctions.h"))
    ms.code.append(C.include(model_header.name))
    ms.code.append(C.include(param_header.name))
    ms.code.append(C.include(weight_header.name))
    if debug:
        ms.code.append(C.include("nn_util.h"))
    else:
        ms.code.append(C.define("ARM_STATUS_CHECK(status)"))
    ms.code.append(C.blank())

    # Write debug variables
    if debug:
        ms.code.extend(outputs.statvars_debug)
        ms.code.append(C.blank())

    # Write static variables
    ms.code.extend(outputs.statvars)
    ms.code.append(C.blank())

    # Write forward pass function
    fw = C.function("nn_forward_pass", typename="int")
    fw.add_param(param_in)
    fw.add_param(param_out)
    ms.code.append(fw)

    # Write forward pass function body
    fw_body = C.block(innerIndent=4)
    fw_body.extend(outputs.function)
    ms.code.append(fw_body)

    mh = C.hfile(model_header)
    mh.code.append(C.blank())
    mh.code.append(C.sysinclude("stdint.h"))
    mh.code.append(C.include("arm_math.h"))
    mh.code.append(C.blank())
    mh.code.append(C.statement(fw))

    with open(param_header, "w") as f:
        f.write(str(ph))
    print("Wrote parameter header:", param_header)

    with open(weight_header, "w") as f:
        f.write(str(wh))
    print("Wrote weight header:", weight_header)

    with open(model_source, "w") as f:
        f.write(str(ms))
    print("Wrote model source code:", model_source)

    with open(model_header, "w") as f:
        f.write(str(mh))
    print("Wrote model header:", model_header)


def cmsis_nn_convert(model_dir, model_name, gen_dir, artifact_dir, debug=False):
    model_path = Path(model_dir)/make_model_filename(model_name)
    transform_path = Path(model_dir)/make_transform_filename(model_name)
    stat_path = Path(model_dir)/make_stats_filename(model_name) 
    dump_path = artifact_dir/f"{model_name}_layers.json"
    param_header = gen_dir/f"{model_name}_params.h"
    weight_header = gen_dir/f"{model_name}_weights.h"
    model_header = gen_dir/f"{model_name}_model.h"
    model_source = gen_dir/f"{model_name}_model.c"

    # FIXME: assume q7_t
    datatype = "q7_t"

    # Convert model
    print(f"Loading model from {model_path}.")
    model = torch.load(model_path)
    print(f"Loading transforms from {transform_path}.")
    transforms = torch.load(transform_path)
    print(f"Loading run stats from {stat_path}.")
    run_stats = torch.load(stat_path)
    print("Converting model parameters.")
    layers = convert_parameters(model, transforms, run_stats, datatype, 0.995)
    print(f"Dumping layer JSON to {dump_path}.")
    dump_layer_json(dump_path, layers)

    # Generate source code/headers
    print("Generating CMSIS-NN code.")
    write_code(layers, model_source, model_header, param_header, weight_header, datatype, debug)
