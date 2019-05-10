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
def create_sb_model(type="alex",  ckpt=None, num_ch2=25, num_ch1=100):

    # Pretraining Model
    if type in {"alex","resnet","googl", "simple"}:
        ae = SplitBrain(encoder=type, num_ch2=num_ch2, num_ch1=num_ch1)
        print_model("Encoder for ch2", ae.ch2_net.encoder)
        print_model("Encoder for ch1", ae.ch1_net.encoder)

        if ckpt != None: # Add ckpt
            file_exists(ckpt)
            print("Loading checkpoint ", ckpt)
            pretrained_dict = torch.load(ckpt, map_location='cpu')
            ae.load_state_dict(pretrained_dict)

    # Finetuning Model
    elif type.split('_')[0] == 'classifier':
        ae = SBNetClassifier(encoder=type.split('_')[1], classifier=type.split('_')[2], num_ch2=num_ch2, num_ch1=num_ch1)
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


'''Determine image display format'''
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

''' Denormalise images of either rgb or lab space from given size of discretization '''
def rescale_color(type, x, bins1, bins2):
    if type == "rgb":
        x[:, 0, :, :] = (x[:, 0, :, :] / bins1)
        x[:, 1, :, :] = (x[:, 1, :, :] / bins2)
        x[:, 2, :, :] = (x[:, 2, :, :] / bins2)
    else:
        x[:, 0, :, :] = x[:, 0, :, :] * 100 / bins1
        x[:, 1, :, :] = (x[:, 1, :, :] * (99. + 87.) / bins2) - 87
        x[:, 2, :, :] = (x[:, 2, :, :] * (95. + 108.) / bins2) - 108

    return x

''' Normalization of LAB space'''
def normalize_lab(x):
    x[0, :, :] = x[0, :, :] / 100.
    x[1, :, :] = (x[1, :, :] + 87.) / (99+87)
    x[2, :, :] = (x[2, :, :] + 108.) / (95+108)
    return x

''' De-Normalization of LAB space'''
def denormalize_lab(x):
    x[:, 0, :, :] = x[:, 0, :, :] * 100.
    x[:, 1, :, :] = (x[:, 1, :, :] * (99.+ 87.)) - 87
    x[:, 2, :, :] = (x[:, 2, :, :] * (95.+ 108.)) - 108
    return x

''' Redundant conversion, useful to work with local image edits/display, else ignore '''
def rgb_to_list(images):
    rgb_imgs = [img for img in images]
    return rgb_imgs

''' Convert LAB space back to RGB '''
def lab_to_rgb_list(images):
    rgb_imgs = [torch.from_numpy(color.lab2rgb(img.permute(1, 2, 0))).permute(2, 0, 1) for img in images]
    return rgb_imgs

''' Normalize RGB images and get a downsampled and discretized version of the image '''
def rgb_preprocess(image, downsample_params=[16,25,100]):

    sample = np.asarray(image) / 255.

    # Get a version of the image that is smaller and with the last dimension as the num_channels (as required by resize library)
    labels = np.array(transform.resize(sample, (downsample_params[0], downsample_params[0]),
                                       preserve_range=True, mode='constant',anti_aliasing=True,
                                       anti_aliasing_sigma=None))

    # Get labels in discrete space
    labels[ :, :, 0] = np.digitize(labels[ :, :, 0], np.linspace(0., 1.01, downsample_params[2]+1)) - 1
    labels[ :, :, 1] = np.digitize(labels[ :, :, 1], np.linspace(0., 1.01, downsample_params[1]+1)) - 1
    labels[ :, :, 2] = np.digitize(labels[ :, :, 2], np.linspace(0., 1.01, downsample_params[1]+1)) - 1

    return sample, labels

''' Convert RGB image to LAB, normalize, and get a downsampled and discretized version of the image '''
def lab_preprocess(image, downsample_params=[16, 25, 100], type="normal"):
    sample = np.transpose(color.rgb2lab(image), (2, 0, 1))  # converts to lab space

    # Get a version of the image that is small and with the last dimension as the num_channels (as required by resize library)
    labels = np.array(transform.resize(np.transpose(sample, (1, 2, 0)), (downsample_params[0], downsample_params[0]),
                                       preserve_range=True, mode='constant', anti_aliasing=True,
                                       anti_aliasing_sigma=None))

    # Get labels in discrete space
    labels[:, :, 0] = np.digitize(labels[:, :, 0], np.linspace(0, 101, downsample_params[2] + 1)) - 1
    # Quantize with full RGB image on LAB space
    if type=="normal":
        labels[:, :, 1] = np.digitize(labels[:, :, 1], np.linspace(-87, 99, downsample_params[1] + 1)) - 1
        labels[:, :, 2] = np.digitize(labels[:, :, 2], np.linspace(-108, 95, downsample_params[1] + 1)) - 1
        # Quantize according to min and max values of the image: Measure of image contrast.
    elif type=="distort":
        labels[:, :, 1] = np.digitize(labels[:, :, 1], np.linspace(np.amin(labels[:, :, 1]), np.amax(labels[:, :, 1]) + 1.,
                                                                   downsample_params[1] + 1)) - 1
        labels[:, :, 2] = np.digitize(labels[:, :, 2], np.linspace(np.amin(labels[:, :, 2]), np.amax(labels[:, :, 2]) + 1.,
                                                               downsample_params[1] + 1)) - 1

    sample = normalize_lab(sample)
    sample = np.transpose(sample, (1, 2, 0))

    return sample, labels

''' For testing purposes: Print info of tensors '''
def print_info(name, sample1, sample2=None):
    print("Info for ", name)
    if sample2==None:
        print("Shapes: ", sample1.shape)
        print("Min vals: ", torch.min(sample1).data)
        print("Max vals: ", torch.max(sample1).data)
        if not isinstance(sample1, torch.LongTensor):
            print("Mean vals: ", torch.mean(sample1).data)
    else:
        print("Shapes: ", sample1.shape, ", ", sample2.shape)
        print("Min vals: ", torch.min(sample1).data, torch.min(sample1).data)
        print("Max vals: ", torch.max(sample2).data, torch.max(sample2).data)
        print("Mean vals: ", torch.mean(sample1).data, torch.mean(sample2).data)
