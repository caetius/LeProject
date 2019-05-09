# Torch
import torch
import torchvision.datasets as datasets
from custom_loader import LABLoader
# Torchvision
import torchvision.transforms as transforms


''' Loads NYU image set with normal ImageLoader '''
def nyu_image_loader(path, batch_size):
    transform = transforms.Compose(
        [
            transforms.ToTensor()
        ]
    )


    sup_train_data = datasets.ImageFolder('{}/{}/train'.format(path, 'supervised'), transform=transform)
    sup_val_data = datasets.ImageFolder('{}/{}/val'.format(path, 'supervised'), transform=transform)
    unsup_data = datasets.ImageFolder('{}/{}/'.format(path, 'unsupervised'), transform=transform)
    data_loader_sup_train = torch.utils.data.DataLoader(
        sup_train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    data_loader_sup_val = torch.utils.data.DataLoader(
        sup_val_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    data_loader_unsup = torch.utils.data.DataLoader(
        unsup_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    return data_loader_sup_train, data_loader_sup_val, data_loader_unsup


''' Loads NYU image set with LabLoader, which turns RGB to LAB before applying the transform '''
def nyu_lab_loader(path, batch_size, downsample_params):

    transform = transforms.Compose(
        [
            transforms.ToTensor()
        ]
    )
    sup_train_data = LABLoader('{}/{}/train'.format(path, 'supervised'), transform=transform, downsample_params=downsample_params)
    sup_val_data = LABLoader('{}/{}/val'.format(path, 'supervised'), transform=transform, downsample_params=downsample_params)
    unsup_data = LABLoader('{}/{}/'.format(path, 'unsupervised'), transform=transform, downsample_params=downsample_params)
    data_loader_sup_train = torch.utils.data.DataLoader(
        sup_train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    data_loader_sup_val = torch.utils.data.DataLoader(
        sup_val_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    data_loader_unsup = torch.utils.data.DataLoader(
        unsup_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    return data_loader_sup_train, data_loader_sup_val, data_loader_unsup


def cifar_image_loader(path='./data', batch_size=16):
    transform = transforms.Compose(
        [transforms.ToTensor(), ])
    trainset = datasets.CIFAR10(root=path, train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=2)
    testset = datasets.CIFAR10(root=path, train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=2)
    # CIFAR-10 Classes
    val_set = testloader

    return trainloader, testloader, val_set

