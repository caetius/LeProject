# Model
from models import SplitBrain, SBNetClassifier

# External
import torchvision
import torch
from torch.autograd import Variable
import numpy as np

import os

# sk-image
from skimage import color
from skimage import transform

file_path = os.path.dirname(os.path.abspath(__file__))


''' Creates model from given params'''
def create_sb_model(type="alex",  ckpt=None, num_ab=25, num_L=100):

    # Pretraining Model
    if type in {"alex","resnet","googl", "simple"}:
        ae = SplitBrain(encoder=type, num_ab=num_ab, num_L=num_L)
        print_model("Encoder for ab", ae.ab_net.encoder)
        print_model("Encoder for L", ae.L_net.encoder)

        if ckpt != None: # Add ckpt
            file_exists(ckpt)
            print("Loading checkpoint ", ckpt)
            pretrained_dict = torch.load(ckpt, map_location='cpu')
            ae.load_state_dict(pretrained_dict)

    # Finetuning Model
    elif type.split('_')[0] == 'classifier':
        ae = SBNetClassifier(encoder=type.split('_')[1], classifier=type.split('_')[2], num_ab=num_ab, num_L=num_L)
        print_model("Pretrained encoder", ae.sp, c=type.split('_')[2], classifier=ae.classifier)

        if ckpt != None: # Add ckpt
            file_exists(ckpt)
            print("Loading checkpoint ", ckpt)
            pretrained_dict = torch.load(ckpt, map_location='cpu')
            ae.sp.load_state_dict(pretrained_dict)
    # Add cuda
    if torch.cuda.is_available():
        ae = ae.cuda()
        print("Model moved to GPU.")
    return ae


''' Print Model '''
def print_model(e, encoder, c=None, classifier=None):
    print("============== %s ==============" % e)
    print(encoder)
    if c != None:
        print("============== %s ==============" % c)
        print(classifier)

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

def grid_imshow(img1, img3, img2=None, second_label="Noised"):
    import matplotlib.pyplot as plt
    plt.figure(figsize= [6,9], dpi=256)
    if isinstance(img2, torch.Tensor) or isinstance(img2, list):
        plt.subplot(311)
        plt.title('Original')
        imshow(torchvision.utils.make_grid(img1))
        plt.subplot(312)
        plt.title(second_label)
        imshow(torchvision.utils.make_grid(img2))
        plt.subplot(313)
        plt.title('Reconstructed')
        imshow(torchvision.utils.make_grid(img3))
    else:
        plt.subplot(211)
        plt.title('Original')
        imshow(torchvision.utils.make_grid(img1))
        plt.subplot(212)
        plt.title('Reconstructed')
        imshow(torchvision.utils.make_grid(img3))
    plt.show()
    plt.savefig(os.path.join(file_path,'images/conv_autoencoder_finetuned.png'))


def normalize(x):
    return x

def denormalize(x):
    return x

def rescale_color(x, c_max):
    x[:,0,:,:] = (x[:,0,:,:] / 100.)
    x[:, 1, :, :] = (x[:, 1, :, :] / 10.)
    x[:, 2, :, :] = (x[:, 2, :, :] /10.)

    return x

def lab_to_rgb(images):
    rgb_imgs = [img for img in images]
    return rgb_imgs

def rgb_to_lab(image, downsample_params=[16,25,100]):

    sample = np.asarray(image) / 255.

    # Get a version of the image that is 16x16 and with the last dimension as the num_channels (as required by resize library)
    labels = np.array(transform.resize(sample, (downsample_params[0], downsample_params[0]), preserve_range=True))

    # Get labels in discrete space
    labels[ :, :, 0] = np.digitize(labels[ :, :, 0], np.linspace(0., 1.01, downsample_params[2]+1)) - 1
    labels[ :, :, 1] = np.digitize(labels[ :, :, 1], np.linspace(0., 1.01, downsample_params[1]+1)) - 1
    labels[ :, :, 2] = np.digitize(labels[ :, :, 2], np.linspace(0., 1.01, downsample_params[1]+1)) - 1

    sample = normalize(sample) # normalized image

    return sample, labels # returns labels as

# Works fine: lab_to_rgb(denormalize(normalize(rgb_to_lab(image))))


def rgb_to_lab_simple(image):
    sample = np.transpose(color.rgb2lab(image),(2,0,1)) # converts to normalized lab space
    sample = normalize(sample) # normalized image
    return np.transpose(sample, (1,2,0)), sample


def print_info(sample1, sample2=None):
    if sample2==None:
        print("Shapes: ", sample1.shape)
        print("Min vals: ", torch.min(sample1))
        print("Max vals: ", torch.max(sample1))
    else:
        print("Shapes: ", sample1.shape, ", ", sample2.shape)
        print("Min vals: ", torch.min(sample1), torch.min(sample1))
        print("Max vals: ", torch.max(sample2), torch.max(sample2))