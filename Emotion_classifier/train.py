from model import CNN
import datahandler as dh
import torch
from torch.autograd  import Variable
from torch.nn import L1Loss, CrossEntropyLoss, BCELoss
from torch.optim import Adam, SGD
from torchsummary import summary
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Current device: ", device)

train_x, train_y = dh.create_batch("./datasets/train_dir/",5,512)
print(train_x.shape)

val_x, val_y = dh.create_batch("./datasets/test_dir/",5,512)
print(val_x.shape)


train_losses = []
val_losses = []