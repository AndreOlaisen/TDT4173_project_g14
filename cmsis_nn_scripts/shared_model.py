from fastai.vision.all import *

# Uses the architecture specs. from the CMSIS-NN CIFAR10 example
# + some BatchNorm layers for testing the folding functionality

def create_model():
    return nn.Sequential(
        # Conv1
        # - 32 x 32 x 3 input
        # - 5 x 5 kernel
        # - 2 padding
        # - 1 stride
        # - 32 x 32 x 32 output
        nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),

        nn.BatchNorm2d(num_features=32),

        # ReLU
        nn.ReLU(),

        # Pool1:
        # - Max. pool
        # - 3 x 3 kernel
        # - 2 stride
        # - 0 padding (this reduces the area in pytorch, need to investigate how cmsis-nn does it...)
        # - 16 x 16 x 32 output
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

        # Conv2:
        # - 16 x 16 x 32 input
        # - 5 x 5 kernel
        # - 2 padding
        # - 1 stride
        # - 16 x 16 x 16 output
        nn.Conv2d(in_channels=32, out_channels=16, kernel_size=5, stride=1, padding=2),

        nn.BatchNorm2d(num_features=16),

        # ReLU
        nn.ReLU(),

        # Pool2:
        # - Max. pool
        # - 3 x 3 kernel
        # - 2 stride
        # - 0 padding
        # - 8 x 8 x 16 output
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

        # Conv3:
        # - 8 x 8 x 16 input
        # - 5 x 5 kernel
        # - 2 padding
        # - 1 stride
        # - 8 x 8 x 32 output
        nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2),

        nn.BatchNorm2d(num_features=32),

        # ReLU
        nn.ReLU(),

        # Pool3:
        # - Max. pool
        # - 3 x 3 kernel
        # - 2 stride
        # - 0 padding
        # - 4 x 4 x 32 output
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

        # FC:
        # - 4 x 4 x 32 input
        # - 10 output
        nn.Flatten(),
        nn.Linear(in_features=4 * 4 * 32, out_features=10),
        nn.BatchNorm1d(num_features=10),

        # SoftMax:
        # - 10 input/output
        nn.Softmax()
)