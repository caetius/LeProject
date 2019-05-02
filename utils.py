# Model
from models import Classifier, get_model

# External
import torch
from torch.autograd import Variable
import numpy as np

import os

''' Instantiate Model '''
def create_model(train_type, ckpt=None, verbose=False, model_type="ae"):
    # Create and print DAE
    if train_type == "pretrain":
        ae = get_model(model_type)
        if ckpt != None:
            file_exists(ckpt)
            print("Loading checkpoint ", ckpt)
            pretrained_dict = torch.load(ckpt, map_location='cpu')
            ae.load_state_dict(pretrained_dict)
        print_model("Encoder", ae.encoder, "Decoder", ae.decoder)
        if torch.cuda.is_available():
            ae = ae.cuda()
            print("Model moved to GPU.")
        return ae
    # Create and print Classifier based on Pretrained DAE
    elif train_type == "classify":
        classifier = Classifier(model_type=model_type, verbose=verbose)
        if ckpt != None:
            file_exists(ckpt)
            print("Loading checkpoint ", ckpt)
            pretrained_dict = torch.load(ckpt, map_location='cpu')
            classifier.ae.load_state_dict(pretrained_dict)
        print_model("Pretrained Encoder", classifier.ae.encoder, "Classifier", classifier.mlp)
        if torch.cuda.is_available():
            classifier = classifier.cuda()
            print("Model moved to GPU.")
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

'''Check if file exists'''
def file_exists(filename):
    if not os.path.isfile(filename):
        raise Exception('The checkpoint file specified does not exist. \n'
                        '(1) check the desired checkpoint name and location\n'
                        '(2) disable ckpt_on (flag)')
        exit(1)


''' Display Image '''
def imshow(img):
    import matplotlib.pyplot as plt
    npimg = img.cpu().numpy()
    plt.axis('off')
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()