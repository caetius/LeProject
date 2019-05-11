# Split Brain Autoencoder (PyTorch)
This is a PyTorch implementation of the [Split-Brain Autoencoders: Unsupervised Learning by Cross-Channel Prediction] (https://arxiv.org/abs/1611.09842) by Zhang et al.
We include working code for RGB, LAB space, and a distorted LAB space prediction, as well as five models including AlexNet, ResNet18, GoogLeNet, and a SimpleNet, consisting of 5 convolutional layers.


## Requirements
- wandb: Track your code on the cloud

- matplotlib + numpy: See input images, noised images, and reconstructed images. 

- torch/torchvision: The model.

- scikit-image: Converting RGB to LAB.


## Arguments
- verbose: Whether to display input, noised input, and output images, plus other additional logs.
- wandb: Name of project on w&b.
- wandb_on: Whether to log runs to w&b.
- model_type: Model identifier string from the list: Model types
- weights_folder: Name of folder where model weights will be saved after every epoch.
- epochs: Number of epochs to train for.
- num_classes_ch1: Num classes for single channel: Generally L (in Lab) or R (in Rgb).
- num_classes_ch1: Num classes for double channel: Generally ab (in Lab) or GB (in rGB).
- downsample_size: Size of downsampled image. Common values are 12x12, 16x16.
- batch_size: Number of images per batch.
- model_type: Model acronym (See below)
- lr_decay: Learning rate decay. Initial learning rate is fixed to 1e-4, but decay can be adapted to decrease by this amount every epoch.
- image_space: The image space of the input and output of the network.

### Finetuning only
- valid: Whether to validate on the validation set.
- num_samples_per_class: Number of samples per class. 

### Pretraining only
- ckpt_on: Whether to load the checkpoint corresponding to current run. This is to continue pretraining on half-pretrained weights, or to use the image display features locally after pretraining.

## Model Acronyms
- alex: Fully Convolutional AlexNet as in the Split Brain paper.
- resnet: ResNet18.
- googl: GoogLeNet.
- simple: A simple net of 5 convolutional layers and batchnorm. ReLU activations.

## Contributing
Please, add new models to model.py. Do not override existing models even if the changes are small.
    
### Contributor Notes
Write here any additional documentation.
