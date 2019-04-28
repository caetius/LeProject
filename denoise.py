import numpy as np
import torch

def corrupt_input(corr_type, data, v):

    if corr_type == 'mask':
        x_corrupted = masking_noise(data, v)

    elif corr_type == 's&p':
        x_corrupted = salt_and_pepper(data, v)

    elif corr_type == 'gauss':
        x_corrupted = gaussian_noise(data, v, 0.1)

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
def salt_and_pepper(X, v):
    rnd = torch.from_numpy(np.random.rand(X.shape[0], X.shape[1], X.shape[2], X.shape[3]))
    noisy = X.clone()
    noisy[rnd < v/2] = 0.
    noisy[rnd > (1 - v/2)] = 1.
    return noisy

'''Apply gaussian noise by adding values sampled from a gaussian to v of the elements in X to the min and max values'''
def gaussian_noise(X, miu, std):
    noise = np.random.normal(loc=miu, scale=std, size=np.shape(X))
    noise_t = torch.from_numpy(noise).type(torch.FloatTensor)
    return torch.add(X,noise_t)