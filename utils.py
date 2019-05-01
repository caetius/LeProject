# Model
from models import Autoencoder, Classifier

# External
import torch
from torch.autograd import Variable
import numpy as np

''' Instantiate Model '''
def create_model(model_type):
    # Create and print DAE
    if model_type == "pretrain":
        ae = Autoencoder()
        print_model("Encoder", ae.encoder, "Decoder", ae.decoder)
        if torch.cuda.is_available():
            ae = ae.cuda()
            print("Model moved to GPU in order to speed up training.")
        return ae
    # Create and print Classifier based on Pretrained DAE
    elif model_type == "classify":
        classifier = Classifier()
        print_model("Pretrained Encoder", classifier.ae.encoder, "Classifier", classifier.mlp)
        if torch.cuda.is_available():
            classifier = classifier.cuda()
            print("Model moved to GPU in order to speed up training.")
        return classifier


''' Print Model '''
def print_model(e, encoder, d, decoder, final=None):
    print("============== %s ==============" % e)
    print(encoder)
    print("============== %s ==============" % d)
    print(decoder)
    if final != None:
        print(final)
    print("")

''' Make var as tensor '''
def get_torch_vars(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

''' Display Image '''
def imshow(img):
    import matplotlib.pyplot as plt
    npimg = img.cpu().numpy()
    plt.axis('off')
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()