import torch
import torch.nn as nn
from fastai.callback.hook import HookCallback
from fastai.learner import load_model

import shared_model


KEY_ACTIVATION_HISTOGRAMS = "activation_histograms"
KEY_BATCHNORM_MEAN_VARS = "batchnorm_mean_vars"


class Log2HistActivationStats(HookCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def hook(self, m, i, o):
        return torch.tensor(o.abs().log2().histc(16, 0, 16))

    def before_fit(self):
        super().before_fit()
        self.hist = []

    def after_batch(self):
        self.hist.append(self.hooks.stored)
        super().after_batch()

    def combined_hist(self):
        return [sum(h[i] for h in self.hist) for i in range(len(self.hist[0]))]


def get_bn_running_mean_vars(model):
    mvs = []
    dev = torch.device("cpu")
    for mod in model.modules():
        if type(mod) == nn.BatchNorm1d or type(mod) == nn.BatchNorm2d:
            mvs.append((mod.running_mean.to(dev), mod.running_var.to(dev)))
    return mvs


def save_run_stats(path, act_hist, bn_mean_vars=[]):
    torch.save({
        KEY_ACTIVATION_HISTOGRAMS: act_hist,
        KEY_BATCHNORM_MEAN_VARS: bn_mean_vars
    }, path)


def load_run_stats(path):
    obj = torch.load(path)
    return obj[KEY_ACTIVATION_HISTOGRAMS], obj.get(KEY_BATCHNORM_MEAN_VARS, [])


def main():
    path = untar_data(URLs.CIFAR)
    dls = ImageDataLoaders.from_folder(path, valid_pct=0.2)
    net = shared_model.create_model()
    learn = Learner(dls, net, loss_func=F.cross_entropy, metrics=accuracy)
    learn.load("cmsis-nn-cifar10-3")
    cb = Log2HistActivationStats()
    learn.validate(cbs=cb)
    print("cb name:", cb.name)
    print("learn has cb:", hasattr(learn, cb.name))
    hist = cb.combined_hist()
    bn_mean_vars = get_bn_running_mean_vars(learn.model)
    save_run_stats("cmsis-nn-cifar10-3-run-stats.pt", hist, bn_mean_vars)


if __name__ == "__main__":
    from fastai.vision.all import *
    main()
