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
            nn.BatchNorm2d(96),
            nn.ReLU(),
			nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 24, 24]
            nn.BatchNorm2d(48),
            nn.ReLU(),
			nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  # [batch, 12, 48, 48]
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),   # [batch, 3, 96, 96]
            nn.Sigmoid(),
        )

    '''Encode then decode'''
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


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
def get_model(type):
    if type == "orig":
        ae = BatchNormAutoencoder()
    elif type == "bn":
        ae = Autoencoder()
    else:
        raise Exception("The model specified does not exist.")
        exit(1)
    return ae