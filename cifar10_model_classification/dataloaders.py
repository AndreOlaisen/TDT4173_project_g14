from torchvision import transforms, datasets
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import typing
import numpy as np
np.random.seed(0)


def get_cifar10_transforms(transfer_learning: bool = False, cmsisnn: bool = False):
    if transfer_learning:
        mean=[0.485, 0.456, 0.406]
        if not cmsisnn:
            std=[0.229, 0.224, 0.225]
        else:
            std = (1.0, 1.0, 1.0)
    else:
        mean = (0.5, 0.5, 0.5)
        if not cmsisnn:
            std = (0.25, 0.25, 0.25)
        else:
            std = (1.0, 1.0, 1.0)
    # Note that transform train will apply the same transform for
    # validation!
    if transfer_learning:
        transform_train = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        transform_test = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else: 
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    return transform_train, transform_test  


def load_cifar10(batch_size: int, validation_fraction: float = 0.1, transfer_learning: bool = False,
                 cmsisnn: bool = False) -> typing.List[torch.utils.data.DataLoader]:
    
    transform_train, transform_test = get_cifar10_transforms(transfer_learning, cmsisnn)

    data_train = datasets.CIFAR10('data/cifar10',
                                  train=True,
                                  download=True,
                                  transform=transform_train)

    data_test = datasets.CIFAR10('data/cifar10',
                                 train=False,
                                 download=True,
                                 transform=transform_test)
 
    indices = list(range(len(data_train)))
    split_idx = int(np.floor(validation_fraction * len(data_train)))

    val_indices = np.random.choice(indices, size=split_idx, replace=False)
    train_indices = list(set(indices) - set(val_indices))

    train_sampler = SubsetRandomSampler(train_indices)
    validation_sampler = SubsetRandomSampler(val_indices)

    dataloader_train = torch.utils.data.DataLoader(data_train,
                                                   sampler=train_sampler,
                                                   batch_size=batch_size,
                                                   num_workers=2,
                                                   drop_last=True)

    dataloader_val = torch.utils.data.DataLoader(data_train,
                                                 sampler=validation_sampler,
                                                 batch_size=batch_size,
                                                 num_workers=2)

    dataloader_test = torch.utils.data.DataLoader(data_test,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=2)

    return dataloader_train, dataloader_test, dataloader_val
