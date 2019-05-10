import torch.utils.data as data
import torch

from PIL import Image

import os
import os.path

from utils import rgb_preprocess, lab_preprocess
import numpy as np

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


''' Helper functions '''

def has_file_allowed_extension(filename, extensions):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)

def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def make_dataset(dir, class_to_idx, extensions):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def default_loader(path):
    return pil_loader(path)

''' Custom Dataloader with three return types: sample, target, downsample'''
class LABLoader(data.Dataset):

    def __init__(self, root, loader=default_loader, extensions=IMG_EXTENSIONS, transform=None, target_transform=None, downsample_params=[16,25,100], image_space="rgb"):
        classes, class_to_idx = find_classes(root)
        samples = make_dataset(root, class_to_idx, extensions)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions
        self.image_space = image_space

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.imgs = self.samples

        self.transform = transform
        self.target_transform = target_transform

        self.downsample_params = downsample_params

    def __getitem__(self, index):

        path, target = self.samples[index]
        sample = self.loader(path)
        cuda = torch.cuda.is_available()

        if self.image_space == "rgb":
            sample, downsample = rgb_preprocess(sample, downsample_params=self.downsample_params)  # Added: Convert to LAB
        elif self.image_space == "lab":
            sample, downsample = lab_preprocess(sample, downsample_params=self.downsample_params, type="normal")  # Added: Convert to LAB
        elif self.image_space == "lab_distort":
            sample, downsample = lab_preprocess(sample, downsample_params=self.downsample_params, type="distort")  # Added: Convert to LAB

        sample = torch.Tensor(np.transpose(sample, (2,0,1))).type(torch.cuda.FloatTensor if cuda else torch.FloatTensor)
        downsample = torch.Tensor(np.transpose(downsample, (2,0,1))).type(torch.cuda.FloatTensor if cuda else torch.FloatTensor)
        return sample, target, downsample

    def __len__(self):
        return len(self.samples)

