import torch
import torch.nn as nn
from collections import namedtuple

# TODO: rename/move

RunStats = namedtuple("RunStats", [
    "act_hist",
    "batchnorm_mv",
    "input_shape"])


class Log2HistActivationStats:
    def __init__(self):
        self.handles = []
        self.mod_idx = {}
        self.hist = []

    def attach(self, model):
        if len(self.handles) > 0:
            self.detach()
        self.mod_idx.clear()
        self.hist.clear()
        for mod in model.modules():
            if hasattr(mod, "weight"):
                handle = mod.register_forward_hook(self.forward_hook)
                self.handles.append(handle)
                self.hist.append(None)
                self.mod_idx[mod] = len(self.hist) - 1

    def detach(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def forward_hook(self, m, i, o):
        """ Calculate log2 histogram of output and add it to
            the existing histograms. Useful for knowing the number
            of bits required to represent the output. """
        hist = o.abs().clamp(min=1.0).log2().ceil().histc(16, 0, 15)
        idx = self.mod_idx[m]
        if self.hist[idx] is None:
            self.hist[idx] = hist
        else:
            self.hist[idx] += hist


def get_batchnorm_mvs(model):
    mvs = []
    for mod in model.modules():
        if type(mod) == nn.BatchNorm1d or type(mod) == nn.BatchNorm2d:
            mvs.append((mod.running_mean, mod.running_var))
    return mvs


def make_model_filename(model_name):
    return f"{model_name}_model.pth"


def make_transform_filename(model_name):
    return f"{model_name}_transforms.pth"


def make_stats_filename(model_name):
    return f"{model_name}_stats.pth"
