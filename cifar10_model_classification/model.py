import argparse
import os
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
from dataloaders import load_cifar10, get_cifar10_transforms
from progressbar import ProgressBar
from model_export import Log2HistActivationStats, get_batchnorm_mvs, RunStats, \
    make_model_filename, make_stats_filename, make_transform_filename


def compute_loss_and_accuracy(
        dataloader: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        loss_criterion: torch.nn.modules.loss._Loss):
    """
    Computes the average loss and the accuracy over the whole dataset
    in dataloader.
    Args:
        dataloder: Validation/Test dataloader
        model: torch.nn.Module
        loss_criterion: The loss criterion, e.g: torch.nn.CrossEntropyLoss()
    Returns:
        [average_loss, accuracy]: both scalar.
    """
    average_loss = 0
    accuracy = 0
    counter = 0
    correct_predictions = 0
    total_size = 0

    with torch.no_grad():
        for (X_batch, Y_batch) in dataloader:
            # Transfer images/labels to GPU VRAM, if possible
            X_batch = utils.to_cuda(X_batch)
            Y_batch = utils.to_cuda(Y_batch)
            # Forward pass the images through our model
            output_probs = model(X_batch)

            # Compute Loss
            average_loss += loss_criterion(output_probs, Y_batch)

            # Compute accuracy
            prediction = output_probs.argmax(dim=1)
            correct_predictions += (prediction == Y_batch).sum().item() 
            total_size += X_batch.shape[0]             
            counter+=1 
    accuracy = correct_predictions/total_size
    average_loss /= counter
    return average_loss, accuracy


def compute_model_stats(
    dataloader: torch.utils.data.DataLoader,
    model: torch.nn.Module):
    """
        Record log2 activation histogram for each layer in the model
        as well as running means and variances for batchnorm layers,
        over the whole dataset in dataloader.
    """

    confusion_matrix = torch.zeros(10, 10, dtype=torch.int64)
    hist_stats = Log2HistActivationStats()
    hist_stats.attach(model)
    with torch.no_grad():
        confusion_matrix = utils.to_cuda(confusion_matrix)
        for X_batch, Y_batch in dataloader:
            X_batch = utils.to_cuda(X_batch)
            pred = model(X_batch).argmax(dim=1)
            for i in range(Y_batch.shape[0]):
                label = Y_batch[i].item()
                pred_label = pred[i].item()
                confusion_matrix[label, pred_label] = confusion_matrix[label, pred_label] + 1
    hist_stats.detach()
    act_hist = hist_stats.hist
    batchnorm_mvs = get_batchnorm_mvs(model)
    return confusion_matrix, act_hist, batchnorm_mvs, X_batch[0].shape


# Model that reaches 77.64% accuracy on test dataset
class Model_1(nn.Module):

    def __init__(self,
                 image_channels,
                 num_classes):
        """
            Is called when model is initialized.
            Args:
                image_channels. Number of color channels in image (3)
                num_classes: Number of classes we want to predict (10)
        """
        super().__init__()
        self.num_filters_cl1 = 32  # Set number of filters in first conv layer
        self.num_filters_cl2 = 64 # "" second conv layer 
        self.num_filters_cl3 = 128 # "" third conv layer
        self.num_filters_cl4 = 256  
        self.num_filters_fcl1 = 64 # number of filter in the first fully connected layer 
        self.num_classes = num_classes # second and last fully connected layer
        # Define the convolutional layers
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=self.num_filters_cl1,
                kernel_size=5,
                stride=1,
                padding=2
                ), 
            nn.ReLU(), 
            nn.Conv2d(in_channels= self.num_filters_cl1,
                    out_channels=self.num_filters_cl2,
                    kernel_size=5,
                    stride=1,
                    padding=2
                ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            nn.Conv2d(in_channels=self.num_filters_cl2, 
                    out_channels=self.num_filters_cl3, 
                    kernel_size=5, 
                    stride=1, 
                    padding=2
                ),
            nn.ReLU(), 
            nn.Conv2d(in_channels=self.num_filters_cl3, 
                    out_channels=self.num_filters_cl4, 
                    kernel_size=5, 
                    stride=1, 
                    padding=2
                ),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=self.num_filters_cl4, 
                    out_channels=self.num_filters_cl4, 
                    kernel_size=5, 
                    stride=1, 
                    padding=2
                ),
            nn.ReLU(), 
            nn.Conv2d(in_channels=self.num_filters_cl4, 
                    out_channels=self.num_filters_cl4, 
                    kernel_size=5, 
                    stride=1, 
                    padding=2
                ),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
       
        # The output of feature_extractor will be [batch_size, num_filters, 4, 4]
        self.num_output_features = 16*self.num_filters_cl4
        # Initialize our last fully connected layer
        # Inputs all extracted features from the convolutional layers
        # Outputs num_classes predictions, 1 for each class.
        # There is no need for softmax activation function, as this is
        # included with nn.CrossEntropyLoss

        # Define the fully connected layers (FCL)
        self.classifier = nn.Sequential(
            nn.Linear(self.num_output_features, self.num_filters_fcl1),
            nn.ReLU(),
            nn.Linear(self.num_filters_fcl1, self.num_classes)
        )

    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """
        batch_size = x.shape[0]
        x = self.feature_extractor(x)
        x = x.view(-1, self.num_output_features)
        x = self.classifier(x)
        out = x
        expected_shape = (batch_size, self.num_classes)
        assert out.shape == (batch_size, self.num_classes),\
            f"Expected output of forward pass to be: {expected_shape}, but got: {out.shape}"
        return out


class Model_2(nn.Module):

    def __init__(self,
                 image_channels,
                 num_classes):
        """
            Is called when model is initialized.
            Args:
                image_channels. Number of color channels in image (3)
                num_classes: Number of classes we want to predict (10)
        """
        super().__init__()
        
        self.full_model = nn.Sequential(
            # Conv1
            # - 32 x 32 x 3 input
            # - 5 x 5 kernel
            # - 2 padding
            # - 1 stride
            # - 32 x 32 x 32 output
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2),

            # nn.BatchNorm2d(num_features=32),

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

            # nn.BatchNorm2d(num_features=16),

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

            # nn.BatchNorm2d(num_features=32),

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
            # nn.BatchNorm1d(num_features=10),

            # SoftMax:
            # - 10 input/output
            # nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """

        return self.full_model(x)


class Trainer:

    def __init__(self,
                 batch_size: int,
                 learning_rate: float,
                 early_stop_count: int,
                 epochs: int,
                 model: torch.nn.Module,
                 dataloaders: typing.List[torch.utils.data.DataLoader],
                 transfer_learning=False):
        """
            Initialize our trainer class.
        """
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.early_stop_count = early_stop_count
        self.epochs = epochs

        # Since we are doing multi-class classification, we use CrossEntropyLoss
        self.loss_criterion = nn.CrossEntropyLoss()
        # Initialize the model
        self.model = model
        # Transfer model to GPU VRAM, if possible.
        self.model = utils.to_cuda(self.model)
        print(self.model)
        # Variable for transfer learning
        self.transfer_learning = transfer_learning 

        # Define our optimizer.
        # If we are using transfer_learning, use the Adam optimizer 
        if self.transfer_learning:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        # SGD = Stochastich Gradient Descent
        else:
            self.optimizer = torch.optim.SGD(self.model.parameters(), self.learning_rate)

        # Load our dataset
        self.dataloader_train, self.dataloader_val, self.dataloader_test = dataloaders
        
        # Validate our model everytime we pass through 50% of the dataset
        self.num_steps_per_val = len(self.dataloader_train) // 2
        self.global_step = 0
        self.start_time = time.time()

        # Tracking variables
        self.VALIDATION_LOSS = collections.OrderedDict()
        self.TEST_LOSS = collections.OrderedDict()
        self.TRAIN_LOSS = collections.OrderedDict()
        self.VALIDATION_ACC = collections.OrderedDict()
        self.TEST_ACC = collections.OrderedDict()

        self.checkpoint_dir = pathlib.Path("checkpoints")

    def validation_epoch(self):
        """
            Computes the loss/accuracy for all three datasets.
            Train, validation and test.
        """
        self.model.eval()
        validation_loss, validation_acc = compute_loss_and_accuracy(
            self.dataloader_val, self.model, self.loss_criterion
        )
        self.VALIDATION_ACC[self.global_step] = validation_acc
        self.VALIDATION_LOSS[self.global_step] = validation_loss
        used_time = time.time() - self.start_time
        print(
            f"Epoch: {self.epoch:>2}",
            f"Batches per seconds: {self.global_step / used_time:.2f}",
            f"Global step: {self.global_step:>6}",
            f"Validation Loss: {validation_loss:.2f},",
            f"Validation Accuracy: {validation_acc:.3f}",
            sep="\t")
        # Compute for testing set
        test_loss, test_acc = compute_loss_and_accuracy(
            self.dataloader_test, self.model, self.loss_criterion
        )
        self.TEST_ACC[self.global_step] = test_acc
        self.TEST_LOSS[self.global_step] = test_loss

        self.model.train()

    def should_early_stop(self):
        """
            Checks if validation loss doesn't improve over early_stop_count epochs.
        """
        # Check if we have more than early_stop_count elements in our validation_loss list.
        if len(self.VALIDATION_LOSS) < self.early_stop_count:
            return False
        # We only care about the last [early_stop_count] losses.
        relevant_loss = list(self.VALIDATION_LOSS.values())[-self.early_stop_count:]
        first_loss = relevant_loss[0]
        if first_loss == min(relevant_loss):
            print("Early stop criteria met")
            return True
        return False

    def train(self, diff_optimizer=False):
        """
        Trains the model for [self.epochs] epochs.
        """
        pbar = ProgressBar()
        
        # Track initial loss/accuracy
        def should_validate_model():
            return self.global_step % self.num_steps_per_val == 0
        print("Starting to train...")
        for epoch in pbar(range(self.epochs)):
            self.epoch = epoch
            # Perform a full pass through all the training samples
            for X_batch, Y_batch in self.dataloader_train:
                # X_batch is the CIFAR10 images. Shape: [batch_size, 3, 32, 32]
                # Y_batch is the CIFAR10 image label. Shape: [batch_size]
                # Transfer images / labels to GPU VRAM, if possible
                X_batch = utils.to_cuda(X_batch)
                Y_batch = utils.to_cuda(Y_batch)

                # Perform the forward pass
                predictions = self.model(X_batch)
                # Compute the cross entropy loss for the batch
                loss = self.loss_criterion(predictions, Y_batch)
                self.TRAIN_LOSS[self.global_step] = loss.detach().cpu().item()

                # Backpropagation
                loss.backward()

                # Gradient descent step
                self.optimizer.step()

                # Reset all computed gradients to 0
                self.optimizer.zero_grad()
                self.global_step += 1
                 # Compute loss/accuracy for all three datasets.
                if should_validate_model():
                    self.validation_epoch()
                    self.save_model()
                    if self.should_early_stop():
                        print("Early stopping.")
                        self.print_outputs()
                        return
        self.print_outputs()

    def save_model(self):
        def is_best_model():
            """
                Returns True if current model has the lowest validation loss
            """
            validation_losses = list(self.VALIDATION_LOSS.values())
            return validation_losses[-1] == min(validation_losses)

        state_dict = self.model.state_dict()
        filepath = self.checkpoint_dir.joinpath(f"{self.global_step}.ckpt")

        utils.save_checkpoint(state_dict, filepath, is_best_model())

    def load_best_model(self):
        state_dict = utils.load_best_checkpoint(self.checkpoint_dir)
        if state_dict is None:
            print(
                f"Could not load best checkpoint. Did not find under: {self.checkpoint_dir}")
            return False
        print("Best model loaded.")
        self.model.load_state_dict(state_dict)
        return True
    
    def print_outputs(self): 
        # Display the final results 
        train_average_loss, train_accuracy = compute_loss_and_accuracy(self.dataloader_train, self.model, self.loss_criterion)
        val_average_loss, val_accuracy = compute_loss_and_accuracy(self.dataloader_val, self.model, self.loss_criterion)
        test_average_loss, test_accuracy = compute_loss_and_accuracy(self.dataloader_test, self.model, self.loss_criterion)
        print("The final average train loss : {}".format(train_average_loss))
        print("The final train accuracy : {}".format(train_accuracy))
        print("The final average validations loss : {}".format(val_average_loss))
        print("The final validation accuracy : {}".format(val_accuracy))
        print("The final average test loss : {}".format(test_average_loss))
        print("The final test accuracy : {}".format(test_accuracy))
        

def create_plots(trainer1: Trainer, name: str):
    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(20, 8))
    plt.subplot(2, 1, 1)
    plt.title("Cross Entropy Loss")
    utils.plot_loss(trainer1.TRAIN_LOSS, label="Training loss")
    utils.plot_loss(trainer1.VALIDATION_LOSS, label="Validation loss")
    utils.plot_loss(trainer1.TEST_LOSS, label="Testing Loss")
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.title("Accuracy")
    utils.plot_loss(trainer1.VALIDATION_ACC, label="Validation Accuracy")
    utils.plot_loss(trainer1.TEST_ACC, label="Testing Accuracy")
    plt.legend()
    plt.savefig(plot_path.joinpath(f"{name}_plot.png"))
    plt.show()
    return


def train_model(model_cls=Model_1, skip_train=False):
    epochs = 10
    batch_size = 64
    learning_rate = 5e-2
    early_stop_count = 4
    dataloaders = load_cifar10(batch_size)
    model = model_cls(image_channels=3, num_classes=10)
    trainer1 = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        model,
        dataloaders
    )
    found = trainer1.load_best_model()
    if not skip_train:
        trainer1.train()
        create_plots(trainer1, model.__class__.__name__)
        return model, dataloaders
    else:
        return (model, dataloaders) if found else (None, None)


def export_model_stats(model, dataloaders, base_path, name_out):
    model_path = base_path/make_model_filename(name_out)
    transform_path = base_path/make_transform_filename(name_out)
    stats_path = base_path/make_stats_filename(name_out)
    _, __, dl_test = dataloaders
    print("Computing model stats on test data.")
    _, act_hist, batchnorm_mvs, in_shape = compute_model_stats(dl_test, model)
    print(f"Saving model to {model_path}.")
    torch.save(nn.Sequential(*list(model.modules())), model_path)
    print(f"Saving transforms to {transform_path}.")
    _, transform_valid = get_cifar10_transforms(transfer_learning=False)
    torch.save(transform_valid, transform_path)
    print(f"Saving model stats to {stats_path}.")
    run_stats = RunStats(act_hist, batchnorm_mvs, in_shape)
    torch.save(run_stats, stats_path)
    return model_path, transform_path, stats_path


if __name__ == "__main__":
    train_model()
