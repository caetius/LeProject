# Torch
import torch
from custom_loader import LABLoader


''' Loads NYU image set with LabLoader, which turns RGB to LAB and also returns a downsampled version of the image '''
def nyu_lab_loader(path, batch_size, downsample_params, image_space):

    sup_train_data = LABLoader('{}/{}/train'.format(path, 'supervised'), transform=None, downsample_params=downsample_params, image_space=image_space)
    sup_val_data = LABLoader('{}/{}/val'.format(path, 'supervised'), transform=None, downsample_params=downsample_params, image_space=image_space)
    unsup_data = LABLoader('{}/{}/'.format(path, 'unsupervised'), transform=None, downsample_params=downsample_params, image_space=image_space)

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



