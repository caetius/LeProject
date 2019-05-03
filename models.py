import torch.nn as nn

'''Define layers of AE for CIFAR-10 or NYU-Dataset input Input'''
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Input size: [batch, 3, 96, 96] (NYU-Dataset)
        # Output size: [batch, 3, 96, 96] (NYU-Dataset)

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),            # [batch, 12, 48, 48]
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),           # [batch, 24, 24, 24]
            nn.ReLU(),
			nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 12, 12]
            nn.ReLU(),
 			nn.Conv2d(48, 96, 4, stride=2, padding=1),           # [batch, 96, 6, 6]
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),  # [batch, 48, 12, 12]
            nn.ReLU(),
			nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 24, 24]
            nn.ReLU(),
			nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  # [batch, 12, 48, 48]
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),   # [batch, 3, 96, 96]
            nn.Sigmoid(),
        )

    '''Encode then decode'''
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

'''Define layers of Batch Norm AE for CIFAR-10 or NYU-Dataset input Input'''
class BatchNormAutoencoder(nn.Module):
    def __init__(self):
        super(BatchNormAutoencoder, self).__init__()
        # Input size: [batch, 3, 96, 96] (NYU-Dataset)
        # Output size: [batch, 3, 96, 96] (NYU-Dataset)

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),            # [batch, 12, 48, 48]
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),           # [batch, 24, 24, 24]
            nn.BatchNorm2d(24),
            nn.ReLU(),
			nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 12, 12]
            nn.BatchNorm2d(48),
            nn.ReLU(),
 			nn.Conv2d(48, 96, 4, stride=2, padding=1),           # [batch, 96, 6, 6]
            nn.BatchNorm2d(96),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),  # [batch, 48, 12, 12]
            nn.BatchNorm2d(48),
            nn.ReLU(),
			nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 24, 24]
            nn.BatchNorm2d(24),
            nn.ReLU(),
			nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  # [batch, 12, 48, 48]
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),   # [batch, 3, 96, 96]
            nn.Sigmoid(),
        )

    '''Encode then decode'''
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

'''Define layers of AE for CIFAR-10 or NYU-Dataset input Input'''
class StackableAE(nn.Module):
    def __init__(self):
        super(StackableAE, self).__init__()
        # Input size: [batch, 3, 96, 96] (NYU-Dataset)

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),            # [batch, 12, 48, 48]
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),           # [batch, 24, 24, 24]
            nn.ReLU(),
			nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 12, 12]
            nn.ReLU(),
 			nn.Conv2d(48, 96, 4, stride=2, padding=1),           # [batch, 96, 6, 6]
            nn.ReLU(),
        )
        # Embedding size: [batch, 3, 96, 96]

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),  # [batch, 48, 12, 12]
            nn.ReLU(),
			nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 24, 24]
            nn.ReLU(),
			nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  # [batch, 12, 48, 48]
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),   # [batch, 3, 96, 96]
            nn.Sigmoid(),
        )
        # Output size: [batch, 3, 96, 96] (NYU-Dataset)


    '''Encode then decode'''
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class StackedAE(nn.Module):
    def __init__(self, model_type, stack_size):
        super(StackedAE, self).__init__()
        self.stack = []
        for i in range(stack_size):
            self.stack.append(get_model(model_type))

    def forward(self, x):
        encoded = x
        for i in range(0,len(self.stack)):
            encoded = self.stack[i].encoder(encoded)
        for i in range(len(self.stack),0,-1):
            decoded = self.stack[i].encoder(decoded)
        return decoded

class CDAutoEncoder(nn.Module):
    r"""
    Convolutional denoising autoencoder layer for stacked autoencoders.
    This module is automatically trained when in model.training is True.
    Args:
        input_size: The number of features in the input
        output_size: The number of features to output
        stride: Stride of the convolutional layers.
    """
    def __init__(self, input_size, output_size, stride):
        super(CDAutoEncoder, self).__init__()

        self.forward_pass = nn.Sequential(
            nn.Conv2d(input_size, output_size, kernel_size=2, stride=stride, padding=0),
            nn.ReLU(),
        )
        self.backward_pass = nn.Sequential(
            nn.ConvTranspose2d(output_size, input_size, kernel_size=2, stride=2, padding=0), 
            nn.ReLU(),
        )

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.1)

    def forward(self, x):
        # Train each autoencoder individually
        x = x.detach()
        # Add noise, but use the original lossless input as the target.
        x_noisy = x * (Variable(x.data.new(x.size()).normal_(0, 0.1)) > -.1).type_as(x)
        y = self.forward_pass(x_noisy)

        if self.training:
            x_reconstruct = self.backward_pass(y)
            loss = self.criterion(x_reconstruct, Variable(x.data, requires_grad=False))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        return y.detach()
    def reconstruct(self, x):
        return self.backward_pass(x)


class StackedAutoEncoder(nn.Module):
    r"""
    A stacked autoencoder made from the convolutional denoising autoencoders above.
    Each autoencoder is trained independently and at the same time.
    """

    def __init__(self):
        super(StackedAutoEncoder, self).__init__()

        self.ae1 = CDAutoEncoder(3, 128, 2)
        self.ae2 = CDAutoEncoder(128, 256, 2)
        self.ae3 = CDAutoEncoder(256, 512, 2)

    def forward(self, x):
        a1 = self.ae1(x)
        a2 = self.ae2(a1)
        a3 = self.ae3(a2)

        if self.training:
            return a3

        else:
            return a3, self.reconstruct(a3)

    def reconstruct(self, x):
            a2_reconstruct = self.ae3.reconstruct(x)
            a1_reconstruct = self.ae2.reconstruct(a2_reconstruct)
            x_reconstruct = self.ae1.reconstruct(a1_reconstruct)
            return x_reconstruct

from collections import OrderedDict
from cytoolz.itertoolz import concat, sliding_window
from typing import Callable, Iterable, Optional, Tuple, List
import torch
import torch.nn as nn


def build_units(dimensions: Iterable[int], activation: Optional[torch.nn.Module]) -> List[torch.nn.Module]:
    """
    Given a list of dimensions and optional activation, return a list of units where each unit is a linear
    layer followed by an activation layer.
    :param dimensions: iterable of dimensions for the chain
    :param activation: activation layer to use e.g. nn.ReLU, set to None to disable
    :return: list of instances of Sequential
    """
    def single_unit(in_dimension: int, out_dimension: int) -> torch.nn.Module:
        unit = [('linear', nn.Linear(in_dimension, out_dimension))]
        if activation is not None:
            unit.append(('activation', activation))
        return nn.Sequential(OrderedDict(unit))
    return [
        single_unit(embedding_dimension, hidden_dimension)
        for embedding_dimension, hidden_dimension
        in sliding_window(2, dimensions)
    ]


def default_initialise_weight_bias_(weight: torch.Tensor, bias: torch.Tensor, gain: float) -> None:
    """
    Default function to initialise the weights in a the Linear units of the StackedDenoisingAutoEncoder.
    :param weight: weight Tensor of the Linear unit
    :param bias: bias Tensor of the Linear unit
    :param gain: gain for use in initialiser
    :return: None
    """
    nn.init.xavier_uniform_(weight, gain)
    nn.init.constant_(bias, 0)


class StackedDenoisingAutoEncoder(nn.Module):
    def __init__(
            self,
            dimensions: List[int],
            activation: torch.nn.Module = nn.ReLU(),
            final_activation: Optional[torch.nn.Module] = nn.ReLU(),
            weight_init: Callable[[torch.Tensor, torch.Tensor, float], None] = default_initialise_weight_bias_,
            gain: float = nn.init.calculate_gain('relu')):
        """
        Autoencoder composed of a symmetric decoder and encoder components accessible via the encoder and decoder
        attributes. The dimensions input is the list of dimensions occurring in a single stack
        e.g. [100, 10, 10, 5] will make the embedding_dimension 100 and the hidden dimension 5, with the
        autoencoder shape [100, 10, 10, 5, 10, 10, 100].
        :param dimensions: list of dimensions occurring in a single stack
        :param activation: activation layer to use for all but final activation, default torch.nn.ReLU
        :param final_activation: final activation layer to use, set to None to disable, default torch.nn.ReLU
        :param weight_init: function for initialising weight and bias via mutation, defaults to default_initialise_weight_bias_
        :param gain: gain parameter to pass to weight_init
        """
        super(StackedDenoisingAutoEncoder, self).__init__()
        self.dimensions = dimensions
        self.embedding_dimension = dimensions[0]
        self.hidden_dimension = dimensions[-1]
        self.activation = activation
        # construct the encoder
        encoder_units = build_units(self.dimensions[:-1], activation)
        encoder_units.extend(build_units([self.dimensions[-2], self.dimensions[-1]], None))
        self.encoder = nn.Sequential(*encoder_units)
        # construct the decoder
        decoder_units = build_units(reversed(self.dimensions[1:]), activation)
        decoder_units.extend(build_units([self.dimensions[1], self.dimensions[0]], final_activation))
        self.decoder = nn.Sequential(*decoder_units)
        # initialise the weights and biases in the layers
        for layer in concat([self.encoder, self.decoder]):
            weight_init(layer[0].weight, layer[0].bias, gain)

    def get_stack(self, index: int) -> Tuple[torch.nn.Module, torch.nn.Module]:
        """
        Given an index which is in [0, len(self.dimensions) - 2] return the corresponding subautoencoder
        for layer-wise pretraining.
        :param index: subautoencoder index
        :return: tuple of encoder and decoder units
        """
        if (index > len(self.dimensions) - 2) or (index < 0):
            raise ValueError('Requested subautoencoder cannot be constructed, index out of range.')
        return self.encoder[index].linear, self.decoder[-(index + 1)].linear

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(batch)



''' Train a classifier consisting of the AE encoder plus a 2-layer MLP and Softmax'''
class Classifier(nn.Module):
    def __init__(self, model_type, verbose):
        super(Classifier, self).__init__()
        self.ae = get_model(model_type) # Select model
        self.mlp = nn.Linear(6*6*96, 1000) # 1000 Classes
        self.verbose = verbose  # Decides whether to also decode images


    # Note that this returns one or two things depending on verbose=True
    def forward(self, x):
        encoded = self.ae.encoder(x)
        mlp = self.mlp(encoded.view(encoded.shape[0], -1))
        if self.verbose:
            decoded = self.ae.decoder(encoded)
            return mlp, decoded
        return mlp


''' This takes a string arg and returns the model corresponding to that string'''
def get_model(type, type2=None, stack=0):
    if type == "orig":
        ae = Autoencoder()
    elif type == "bn":
        ae = BatchNormAutoencoder()
    elif type == "sae":
        ae = StackedAutoEncoder()
    elif type == "sdae_st1":
        ae = StackedDenoisingAutoEncoder([96, 500, 500, 2000, 10])
    elif type == "sdae_st2":
        ae = StackedDenoisingAutoEncoder([96,1000,1000,1000])
    elif type == "stacked":
        ae = StackedAE(model_type=type2, stack_size=stack)
    else:
        raise Exception("The model specified does not exist.")
        exit(1)
    return ae
