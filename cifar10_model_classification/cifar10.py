import argparse
import pathlib
import importlib
import json

import utils
from model import train_model, export_model_stats
from cmsisnn.convert import cmsis_nn_convert
from cmsisnn.external import cmake_build
from cmsisnn.evaluate import evaluate_generated
from cmsisnn.analyze import plot_activation_histograms


CMAKE_ROOT = (pathlib.Path(__file__).parent/".."/"cmsis_nn").absolute()
BUILD_PATH = CMAKE_ROOT/"build"
SOURCE_PATH = CMAKE_ROOT/"generated"
SOURCE_GEN_PATH = SOURCE_PATH/"model"
EXECUTABLE_PATH = BUILD_PATH/"generated"/"generated"
EVAL_PATHS = list(pathlib.Path("~/.fastai/data/cifar10/test").expanduser().iterdir())
EVAL_COUNT = 1000


def get_model_class(class_name):
    module = importlib.import_module("model")
    model_class = getattr(module, class_name)
    return model_class


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train", action="store_true")
    p.add_argument("--model", default="Model_1")
    p.add_argument("--name", default=None)
    p.add_argument("--export", action="store_true")
    p.add_argument("--convert", action="store_true")
    p.add_argument("--build", action="store_true") 
    p.add_argument("--eval", default="")
    p.add_argument("--clean", action="store_true")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.train or args.export:
        try:
            model_class = get_model_class(args.model)
        except (ImportError, AttributeError):
            raise RuntimeError(f"Couldn't find model model.{args.model}!")
        model, dataloaders = train_model(model_class, not args.train)
        if model is None or dataloaders is None:
            raise RuntimeError("Couldn't load model checkpoint!")
    export_name = args.model if args.name is None else args.name
    export_dir = utils.create_export_path("model")
    if args.export:
        export_model_stats(model, dataloaders, export_dir, export_name)
    if args.convert:
        artifact_dir = utils.create_export_path("convert")
        cmsis_nn_convert(export_dir, export_name, SOURCE_GEN_PATH, artifact_dir, args.debug)
    if args.build:
        cmake_build(CMAKE_ROOT, clean=args.clean)
    if args.eval:
        run_dir = utils.create_export_path("eval")
        eval_path = pathlib.Path(args.eval)
        correct_count = 0
        total_count = 0
        for p in EVAL_PATHS:
            label = p.name
            print("Evaluating", label, end=" ... ", flush=True)
            counts, ties = evaluate_generated(EXECUTABLE_PATH, p,
                                              label, EVAL_COUNT, run_dir)
            label_count = counts[label]
            sum_count = sum(counts.values())
            label_accuracy = label_count / sum_count
            correct_count += label_count
            total_count += sum_count
            print(f"Accuracy: {label_accuracy:.4g}")
        accuracy = correct_count / total_count
        print(f"Overall accuracy: {accuracy:.4g}")
        """
        layer_json = utils.create_export_path("convert")/f"{export_name}_layers.json"
        print(f"Reading activations from {output_json}.")
        with open(output_json, "r") as f:
            activations = json.load(f)
        print(f"Reading layers from {layer_json}.")
        with open(layer_json) as f:
            layers = json.load(f)
        plot_activation_histograms(activations, layers)
        """

if __name__ == "__main__":
    main()
