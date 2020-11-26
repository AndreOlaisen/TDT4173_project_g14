# CMSIS-NN demonstration

## Directory contents

* **android**: Android application for displaying the CIFAR-10 class scores from the nRF9160.
* **CMSIS_5**: [CMSIS 5 repository](https://github.com/ARM-software/CMSIS_5) clone.
* **generated**: Example application for running generated CMSIS-NN model on PC.
* **zephyr**: nRF Connect projects for running generated CMSIS-NN model on nRF9160.
* **scripts**: Contains script for sending image data to device over serial and receiving class scores.
* **util**: Debug code for dumping layer activation data to JSON.

## Requirements for the PC example

* (Git)
* GCC
* CMake

## Setup/build for the PC example

1. Generate CMSIS-NN model from PyTorch to `generated/model`, see [here](../cifar10_model_classification/README.md).
1. Make sure that the CMSIS_5 submodule is cloned and apply a patch to make it build on the host. From the repository root, run:
    * `git submodule init`
    * `git submodule update`
    * `cd cmsis_nn/CMSIS_5`
    * `git apply ../cmsis-host.diff`
2. Create a CMake build directory and generate build files:
    * `mkdir build && cd build`
    * `cmake ..`
3. Build the generated example:
    * `cmake --build .`

## Running the PC example

The executable is built to `build/generated/generated`.

To test the program, run `build/generated/generated path/to/image.png`.
The `path/to/image.png` should be a path to a 32x32 image from the CIFAR-10 dataset in .png format.

The program prints the predicted scores for each class, for example:
```
% build/generated/generated cifar10/train/bird/12361_bird.png

airplane: 5
automobile: -2
bird: 25
cat: -3
deer: -11
dog: 4
frog: 14
horse: -12
ship: -8
truck: -18
```