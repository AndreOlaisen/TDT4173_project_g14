#!/usr/bin/env python3

import json
import pathlib
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt

from cmsisnn.convert import QFormat


def get_layer_act_format(layers, name):
    for l in layers:
        if l["name"] == name:
            return QFormat(*l["parameters"]["out_qformat"])
    raise KeyError()


def parse_int8_str(val, qformat=None):
    signed = int.from_bytes(bytes([int(val, 16)]), "little", signed=True)
    if qformat is not None:
        return float((signed / 2**qformat.fbits))
    else:
        return signed


def plot_histogram(axes, act_values, label):
    hist, edges = np.histogram(act_values, bins=256,
                               range=(-128, 127), density=True)
    min_pct = hist[0] * 100.0
    max_pct = hist[-1] * 100.0
    zero_pct = sum(1 for a in act_values if a == 0) / sum(act_values)
    text = f"Min: {min_pct:.2g}%, Max: {max_pct:.2g}%, Z: {zero_pct:.2g}%"
    axes.bar(list(range(-128, 127 + 1)), hist, width=1.0, label=label)
    props = dict(boxstyle='round', facecolor='gray', alpha=0.5)
    axes.text(0.75, 0.90, text, transform=axes.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)


def plot_activation_histograms(activations, show=True, save_path=None):
    fig, axes = plt.subplots(len(activations), 1, sharex=True, tight_layout=True)
    for i, act in enumerate(activations):
        plot_histogram(axes[i], act["activations"], act["name"])
        axes[i].legend(loc="upper left")
    if show:
        plt.show()
    if save_path is not None:
        plt.savefig(save_path)


def load_activation_json(path):
    with open(path, "r") as f:
        activations = json.load(f)
    for obj in activations:
        obj["activations"] = [parse_int8_str(a) for a in obj["activations"]]
    return activations


def main():
    p = argparse.ArgumentParser()
    p.add_argument("activations")
    args = p.parse_args()
    activations = load_activation_json(args.activations)
    plot_activation_histograms(activations)


if __name__ == "__main__":
    main()
