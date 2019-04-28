# Denoising AutoEncoder (PyTorch)
This is an implementation parting from the denoising autoencoder from Vincent et al. 2008.

We extend the work by using convolutional layers and stacking the autoencoders. 

Similar to the original paper, we support masking noise. Two additional modes of noising have been added:
1. Salt-and-pepper noise
2. Gaussian noise

#Next Steps:
1. Choice of network depth and num. hidden layers.
2. Explore different loss functions.
3. Explore all three noising strategies. 
4. Try Stacked autoencoder.