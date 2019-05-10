import numpy as np
from skimage.color import rgb2lab

'''
This class checks the range of LAB values returned when running skimage conversion of RGB on your machine. If the range is different to that 
seen in utils.py, then you must change this. The default values used in our code are illuminant="D65" and observer="2", as per the sk-image defaults. 
'''

def get_lab_range():
    print("Generating all RGB pixels")

    image = np.zeros((4096,4096,3))
    i = 0
    j = 0

    maxL = 0.;
    minL = 100.;

    maxA = -127.;
    maxB = -127.;
    minA = 128.;
    minB = 128.;

    for r in range(256):
        for g in range(256):
            for b in range(256):
                image[i,j] = np.array([r/255.,b/255.,g/255.])
                i+=1
                if i == 4096:
                    i=0
                    j+=1

    print("Image generated: ", image.shape)
    lab_image = rgb2lab(image)
    print("image converted: ", lab_image.shape)
    maxL = max(np.amax(lab_image[:,:,0]), maxL)
    minL = min(np.amin(lab_image[:,:,0]), minL)
    maxA = max(np.amax(lab_image[:,:,1]), maxA)
    minA = min(np.amin(lab_image[:,:,1]), minA)
    maxB = max(np.amax(lab_image[:,:,2]), maxB)
    minB = min(np.amin(lab_image[:,:,2]), minB)

    print("Range L : ", minL, ", ", maxL)
    print("Range A : ", minA, ", ", maxA)
    print("Range B : ", minB, ", ", maxB)

get_lab_range()