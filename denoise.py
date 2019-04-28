import numpy as np
import torch

def corrupt_input(corr_type, data, v):
    """ Corrupt a fraction 'v' of 'data' according to the
    noise method of this autoencoder.
    :return: corrupted data
    """

    if corr_type == 'mask':
        x_corrupted = masking_noise(data, v)

    elif corr_type == 's&p':
        x_corrupted = salt_and_pepper(data, v)

    elif corr_type == 'gauss':
        x_corrupted = gaussian_noise(data, v, 0.2)

    elif corr_type == 'none':
        x_corrupted = data

    else:
        x_corrupted = None

    return x_corrupted


'''Apply masking noise by zeroing out a fraction v of the elements in X'''
def masking_noise(X, v):
    noise_tensor = (torch.from_numpy(np.random.uniform(0,1,X.shape)) > v).type(torch.FloatTensor)
    return torch.mul(X, noise_tensor)

'''Apply salt and pepper noise by setting a fraction v of the elements in X to the min and max values'''
def salt_and_pepper(img, v):
    rnd = np.random.rand(img.shape)
    noisy = img[:]
    noisy[rnd < v/2] = 0.
    noisy[rnd > 1 - v/2] = 1.
    return noisy

'''TODO: Check if this works - Apply gaussian noise by adding values sampled from a gaussian to v of the elements in X to the min and max values'''
def gaussian_noise(X, miu, std):
    noise = np.random.normal(loc=miu, scale=std, size=np.shape(X))
    noise_t = torch.from_numpy(noise)
    return X+noise_t # Normalize?