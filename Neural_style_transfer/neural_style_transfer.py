from __future__ import print_function
import PIL
from matplotlib import image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy

#################*********************#################

#check for the available device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

# desired size of the output image
imsize = 512 if torch.cuda.is_available() else 128  # use small size if no gpu

loader = transforms.Compose([
    transforms.Resize(imsize),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor


def image_loader(image_name):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


style_img = image_loader("C:\\Users\\User\\Desktop\\Strive_School\\Github\\Projects\\Neural_style_transfer\\images\\picasso.jpg")
content_img = image_loader("C:\\Users\\User\\Desktop\\Strive_School\\Github\\Projects\\Neural_style_transfer\\images\\dancing.jpg")

assert style_img.size() == content_img.size(), \
    "we need to import style and content images of the same size"


unloader = transforms.ToPILImage()  # reconvert into PIL image

#plt.ion() # turn on interactive mode

def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)

    if title is not None:
        plt.title(title)
    #plt.pause(5) # pause a bit so that plots are updated

    return image


#find another way to show the images if you have time
a = imshow(style_img, title='Style Image')
a.show()

b = imshow(content_img, title='Content Image')
b.show()