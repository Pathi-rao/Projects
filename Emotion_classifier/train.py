from model import CNN
import datahandler as dh
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch import optim
from torchsummary import summary
from sklearn.metrics import accuracy_score, confusion_matrix


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Current device: ", device)
epochs = 100
batch_size = 32 


train_loader, test_loader = dh.pre_processor(
                                'C:\\Users\\User\\Desktop\\Strive_School\\Github\\Datasets\\archive\\fer2013\\',
                                batch_size)

#print(len(train_loader), len(test_loader))

model = CNN() # base model that we created

(summary(model, (1, 48, 48))) # prints the summary of the model. 48*48 is our input image size

#define the optimizer and the loss
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
#scheduler = ReduceLROnPlateau(optimizer, 'max', factor = 0.5, verbose=True)
# criterion = nn.NLLLoss()
criterion = nn.CrossEntropyLoss()



epochs = 3
print_every = 40

train_loss = []
val_loss = []
iters  = [] # save the iteration counts here for plotting
#losses = [] # save the avg loss here for plotting

for e in range(epochs):
    running_loss = 0
    print(f"Epoch: {e+1}/{epochs}")

    for i, (images, labels) in enumerate(iter(train_loader)):
        
        # print('oh yeah')
        # actual = (labels < 3).reshape([1,1]).type(torch.FloatTensor)
        # print(actual)

        # reset the grdients
        optimizer.zero_grad()
        
        output = model(images)   # 1) Forward pass
        loss = criterion(output, labels) # 2) Compute loss
        loss.backward()                  # 3) Backward pass
        optimizer.step()                 # 4) Update model
        
        running_loss += loss.item()
        
        if i % print_every == 0:
            print(f"\tIteration: {i}\t Loss: {running_loss/print_every:.4f}")
            running_loss = 0
    train_loss.append(running_loss/images.shape[0])

    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for i, (images, labels) in enumerate(iter(test_loader)):
            outputs = model(images)
            v_loss = criterion(outputs, labels)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        val_loss.append(v_loss/images.shape[0])

    print('Accuracy of the network on test images: %d %%' % (
        100 * correct / total))