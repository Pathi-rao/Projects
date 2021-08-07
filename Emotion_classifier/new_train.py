from model import CNN
import datahandler as dh
import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torchsummary import summary
from sklearn.metrics import accuracy_score, confusion_matrix


train_loader, test_loader = dh.pre_processor(
                                'C:\\Users\\User\\Desktop\\Strive_School\\Github\\Datasets\\archive\\fer2013\\',
                                batchsize = 32)


train_losses = []
val_losses = []
val_loss = []

def train(epochs, train_loader, model, optimizer, criterion, scheduler, best_val, print_every = 40):

    model.train()

    for each_epoch in range(epochs):

        tr_loss = 0

        for i , (images, labels) in enumerate(iter(train_loader)):
            images = images.to(device)
            labels = labels.to(device)
            
            # print(type(images))
            optimizer.zero_grad()

            output_train = model(images)

            loss_train = criterion(output_train, labels)

            loss_train.backward()

            optimizer.step()
            # print(loss_train.item())
            
            tr_loss += loss_train.item()

        if i % print_every == 0:
            # print(f"\tIteration: {i}\t Loss: {tr_loss/print_every:.4f}")
            train_losses.append(tr_loss/images.shape[0])
            print('Epoch : ',each_epoch, "\t Train loss: ", tr_loss/images.shape[0])

            val_loss = 0
            with torch.no_grad():
                for i , (val_images, val_labels) in enumerate(iter(test_loader)):
                    output = model(val_images.to(device))
                    loss_val = criterion(output, val_labels.to(device))
                    val_loss += loss_val.item()


            val_loss =  val_loss/val_images.shape[0]
            val_losses.append(val_loss)
            print('Epoch : ',each_epoch, "\t Train loss: ", tr_loss/val_images.shape[0], "\t Validation loss: ", val_loss)

            if val_loss < best_val:
                best_val = val_loss
                torch.save(model, "./models/best_model.pth")
        scheduler.step()
    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.show()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Current device: ", device)

model = CNN()

# defining the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0005)
scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)

# defining the loss function
criterion = nn.CrossEntropyLoss()


# summary(model,(1, 48, 48)) # since the input images are of gray sclae size 48*48

train(epochs = 5, train_loader = train_loader, model = model, optimizer = optimizer,
        criterion = criterion, scheduler = scheduler, best_val = 2, print_every = 40)