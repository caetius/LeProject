# Denoising AutoEncoder (PyTorch)
This is a PyTorch implementation parting from the denoising autoencoder with convolutional layers in Masci et al. 2011, 
an extension of the denoising autoencoder of Vincent et al. 2008, and the stacked denoising autoencoder of Vincent et al. 2010.

## Requirements
wandb: Track your code on the cloud

matplotlib + numpy: See input images, noised images, and reconstructed images. 


torch/torchvision: The model.


## Arguments
- valid: Whether to validate on the validation set.
- perc_noise: Percentage of pixels to corrupt in image. In the case of Gaussian noise, it is the standard deviation of the zero-mean noise.
- corr_type: Noise type (corruption, see below)
- verbose: Whether to display input, noised input, and output images, plus other additional logs.
- wandb: Name of project on w&b.
- wandb_on: Whether to log runs to w&b.
- model_type: Model identifier string from the list: Model types

#####Classifier only
- add_noise: Whether to add noise to the finetuning stage. This is discouraged by Vincent et al. 2010.

#####Pretraining only
- ckpt_on: Whether to load the checkpoint corresponding to current run.


## Noise Types
1. Salt-and-pepper noise (push lowest values to zero, highest to one)
2. Gaussian noise (add perturbation sampled from zero-mean Gaussian)
3. Masking noise (zero out a percentage of pixels)

## Model Types
- orig: Our first model version.
- bn: orig + batch norm.

##Contributing
Please, add new models to model.py. Do not override existing models even if the changes are small. Add your model to the keys of get_model() in model.py (please give your model a reasonable shorthand).


##Next Steps:
1. Aishwarya: Batch norm experiments.
    - For all three noising types and for noise levels (5%,10%,20%50%)
2. Lekha: Develop a stacked version of the autoencoder.
    - Run with and without batch norm
3. Diego: Find bug in the current finetuning. 
    - Re-run all current experiments, add zero-noise experiment.
    - Get code in format needed to submit tomorrow.
    

### Contributor Notes
Write here any additional documentation.
