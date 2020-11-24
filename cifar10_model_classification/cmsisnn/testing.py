import torch
import torch.nn as nn
import torchvision.transforms
import utils
import pathlib
import json
from PIL import Image
from cmsis_nn_convert import Converter, write_code, quantize
from cmsis_nn_run import cmake_build, generated_run
from cmsis_nn_analyze import load_activation_json
from model_export import RunStats

torch.manual_seed(0)
torch.set_deterministic(True)


def randn_quantizable(size):
    data = torch.randn(size)
    data /= data.abs().sum().item()
    data = (data * 2**7).round() / 2**7
    return data


def unquantize(data, fbits):
    return (data / 2**fbits)


def find_activation(activations, name):
    for obj in activations:
        if obj["name"] == name:
            return obj["activations"]
    raise KeyError("Unable to find layer in activation record.")


def flat_hwc_to_chw(data, shape):
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data)
    if len(shape) == 4:
        data = data.reshape(shape[0], shape[2], shape[3], shape[1])
        data = data.permute(0, 3, 1, 2)
    else:
        data = data.reshape(shape[1], shape[2], shape[0])
        data = data.permute(2, 0, 1)
    return data


def test_conv2d():
    # Create output image
    export_path = utils.create_export_path("test")
    input_data = torch.randn(3, 2, 2)
    image = torchvision.transforms.functional.to_pil_image(input_data, mode="RGB")
    image_path = export_path/"conv_test.png"
    image.save(image_path)

    layer = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3,
                      stride=1, padding=1)
    layer.requires_grad_(False)
    layer.weight.data = randn_quantizable(layer.weight.data.shape)
    layer.bias.data = randn_quantizable(layer.bias.data.shape)

    image = Image.open(image_path)
    transforms = [torchvision.transforms.ToTensor(),
                  torchvision.transforms.Normalize([0.5] * 3, [1.0] * 3)]
    input_data = transforms[0](image)
    print("Image as tensor\n:", input_data)
    input_data = transforms[1](input_data)
    print("Input normalized\n:", input_data)
    input_data = input_data.reshape(1, *input_data.shape)
    output_data = layer(input_data)

    act_hist_conv = output_data.abs().clamp(min=1.0).log2().ceil().histc(16, 0, 15)
    act_hist = [act_hist_conv]
    run_stats = RunStats(act_hist=act_hist, batchnorm_mv=[], input_shape=(3, 2, 2))
    converter = Converter(transforms, run_stats, "q7_t")
    converter.convert_conv2d(layer, converter.layers[-1])
    cmsis_layers = converter.layers
    root_path = pathlib.Path("../cmsis_nn")
    base_path = root_path/"generated"/"model"
    base_name = "conv2d_test"
    write_code(cmsis_layers, base_path/f"{base_name}.c", base_path/f"{base_name}.h",
               base_path/f"{base_name}_params.h", base_path/f"{base_name}_weights.h",
               "q7_t", debug=True)
    cmake_build(root_path, clean=True)
    activation_path = generated_run(root_path, export_path, image_path)
    activations = load_activation_json(activation_path)
    input_act_values = find_activation(activations, "INPUT0")
    input_act_reshaped = flat_hwc_to_chw(input_act_values, input_data.shape)
    input_quant, qformat = quantize(input_data, 7)
    print("Input data:\n", input_data)
    print("Got input data:\n", unquantize(input_act_reshaped, 7))
    print(f"Input quantized: {input_quant.shape} {qformat}\n", input_quant)
    print(f"Got input quantized: {input_act_reshaped.shape}\n", input_act_reshaped)
    conv_act_values = find_activation(activations, "CONV_RGB0")
    conv_act_reshaped = flat_hwc_to_chw(conv_act_values, output_data.shape)
    output_quant, qformat = quantize(output_data, 7)
    print(f"Expected: {output_quant.shape}\n")
    print(output_quant)
    print(f"Got: {conv_act_reshaped.shape}\n")
    print(conv_act_reshaped)


def test_linear():
    # Create output image
    export_path = utils.create_export_path("test")
    input_data = torch.randn(3, 2, 2)
    image = torchvision.transforms.functional.to_pil_image(input_data, mode="RGB")
    image_path = export_path/"linear_test.png"
    image.save(image_path)

    flat = nn.Flatten()
    layer = nn.Linear(in_features=3 * 2 * 2, out_features=4, bias=False)
    flat.requires_grad_(False)
    layer.requires_grad_(False)
    weight = torch.arange(1, int(layer.in_features * layer.out_features) + 1).float()
    # bias = torch.arange(1, int(layer.out_features) + 1).float()
    layer.weight.data = (weight / 100.0).reshape(layer.weight.data.shape)
    # layer.bias.data = (bias / 10.0).reshape(layer.bias.data.shape)

    print("Weight shape:", layer.weight.data.shape)

    image = Image.open(image_path)
    transforms = [torchvision.transforms.ToTensor(),
                  torchvision.transforms.Normalize([0.5] * 3, [1.0] * 3)]
    input_data = transforms[0](image)
    print("Image as tensor\n:", input_data)
    input_data = transforms[1](input_data)
    print("Input normalized\n:", input_data)
    input_data = input_data.reshape(1, *input_data.shape)
    output_data = flat(input_data)
    output_data = layer(output_data)

    act_hist_conv = output_data.abs().clamp(min=1.0).log2().ceil().histc(16, 0, 15)
    act_hist = [act_hist_conv.tolist()]
    print(act_hist)
    run_stats = RunStats(act_hist=act_hist, batchnorm_mv=[], input_shape=(3, 2, 2))
    converter = Converter(transforms, run_stats, "q7_t")
    converter.convert_linear(layer, converter.layers[-1])
    cmsis_layers = converter.layers
    root_path = pathlib.Path("../cmsis_nn")
    base_path = root_path/"generated"/"model"
    base_name = "linear_test"
    write_code(cmsis_layers, base_path/f"{base_name}.c", base_path/f"{base_name}.h",
               base_path/f"{base_name}_params.h", base_path/f"{base_name}_weights.h",
               "q7_t", debug=True)
    cmake_build(root_path, clean=True)
    activation_path = generated_run(root_path, export_path, image_path)
    activations = load_activation_json(activation_path)
    input_act_values = find_activation(activations, "INPUT0")
    input_act_reshaped = flat_hwc_to_chw(input_act_values, input_data.shape)
    input_quant, qformat = quantize(input_data, 7)
    print("Input data:\n", input_data)
    print("Got input data:\n", unquantize(input_act_reshaped, 7))
    print(f"Input quantized: {input_quant.shape} {qformat}\n", input_quant)
    print(f"Got input quantized: {input_act_reshaped.shape}\n", input_act_reshaped)
    fc_act_values = torch.tensor(find_activation(activations, "FC0"))
    output_quant, qformat = quantize(output_data, 7)
    print("Output data:\n", output_data)
    print(f"Expected: {output_quant.shape} {qformat}")
    print(output_quant)
    print(f"Got: {fc_act_values.shape}")
    print(fc_act_values)


# Next to check: max pooling (weird padding + dilation)


if __name__ == "__main__":
    # test_conv2d()
    test_linear()
