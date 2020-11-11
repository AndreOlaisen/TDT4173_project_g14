import pathlib
import matplotlib.pyplot as plt
import torch
import utils
import time
import typing
import torchvision
import collections
from torch import nn
from tqdm import tqdm
from model import Trainer 
from dataloaders import load_cifar10

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(512, 10) # Replacing the 1000-layer softmax with a fully-connected layer with 10 outputs
        # No need to apply softmax, as this is done in nn.CrossEntropyLoss

        for param in self.model.parameters(): # Freeze all parameters
            param.requires_grad = False
        for param in self.model.fc.parameters(): # Unfreeze the last fully-connected
            param.requires_grad = True # layer
        for param in self.model.layer4.parameters(): # Unfreeze the last 5 convolutional
            param.requires_grad = True # layers
    
    def forward(self, x):
        x = self.model(x)
        return x

def create_plots(trainer1: Trainer, name: str):
    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(20, 8))
    # The loss 
    plt.subplot(2, 1, 1)
    plt.title("Cross Entropy Loss")
    utils.plot_loss(trainer1.TRAIN_LOSS, label="Training loss")
    utils.plot_loss(trainer1.VALIDATION_LOSS, label="Validation loss")
    utils.plot_loss(trainer1.TEST_LOSS, label="Testing Loss")
    plt.legend()
    plt.subplot(2, 1, 2)
    # The accuracy 
    plt.title("Accuracy")
    utils.plot_loss(trainer1.VALIDATION_ACC, label="Validation Accuracy")
    utils.plot_loss(trainer1.TEST_ACC, label="Testing Accuracy")
    plt.legend()
    plt.savefig(plot_path.joinpath(f"{name}_plot.png"))
    plt.show()
    return 

if __name__ == "__main__":
    epochs = 10
    batch_size = 32
    learning_rate = 5e-4
    early_stop_count = 4
    dataloaders = load_cifar10(batch_size, transfer_learning=True)
    model = Model()
    trainer1 = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        model,
        dataloaders,
        transfer_learning=True
    )
    trainer1.train()
    create_plots(trainer1, "Model_transfer_learning")
