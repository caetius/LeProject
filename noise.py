import torch

def corrupt_input(corr_type, data, v):

    if corr_type == 'mask':
        x_corrupted = masking_noise(data, v)

    elif corr_type == 'sp':
        x_corrupted = salt_and_pepper(data, v)

    elif corr_type == 'gauss':
        x_corrupted = gaussian_noise(data, 0, v)

    elif corr_type == 'none':
        x_corrupted = data

    else:
        x_corrupted = None

    return x_corrupted


'''Apply masking noise by zeroing out a fraction v of the elements in X'''
def masking_noise(X, v):
    noise_tensor = (torch.distributions.uniform.Uniform(0, 1).sample(X.shape) > v).type(torch.FloatTensor)
    if torch.cuda.is_available():
        noise_tensor = noise_tensor.type(torch.cuda.FloatTensor)
    return torch.mul(X, noise_tensor)

'''Apply salt and pepper noise by setting a fraction v of the elements in X to the min and max values'''
def salt_and_pepper(X, v):
    rnd = torch.distributions.uniform.Uniform(0, 1).sample(X.shape)
    if torch.cuda.is_available():
        rnd = rnd.type(torch.cuda.FloatTensor)
    noisy = X.clone()
    noisy[rnd < v/2] = 0.
    noisy[rnd > (1 - v/2)] = 1.
    return noisy

'''Apply gaussian noise by adding values sampled from a gaussian to v of the elements in X to the min and max values'''
def gaussian_noise(X, miu, std):
    noise = torch.distributions.normal.Normal(miu, std).sample(X.shape)
    if torch.cuda.is_available():
        noise = noise.type(torch.cuda.FloatTensor)
    return torch.clamp(torch.add(X,noise),0,1)