""" 
    let’s run a simple distributed training with two clients and one server. Our training procedure 
    and network architecture are based on PyTorch’s Deep Learning with PyTorch.
"""

# import Flower and PyTorch related packages
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10

import flwr as fl

# check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data():
    """Load CIFAR-10 (training and test set)."""
    transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = CIFAR10(".", train=True, download=True, transform=transform)
    testset = CIFAR10(".", train=False, download=True, transform=transform)

    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    testloader = DataLoader(testset, batch_size=32, shuffle=False)

    return trainloader, testloader


""" Define the loss and optimizer with PyTorch. The training of the dataset is done by looping over the
    dataset, measure the corresponding loss and optimize it.
"""
def train(net, trainloader, epochs):
    """Train the network on the training set."""

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()

"""
    Define then the validation of the machine learning network. We loop over the test set and measure 
    the loss and accuracy of the test set.
"""
def test(net, testloader):
    """Validate the network on the entire test set."""

    criterion = nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0

    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    return loss, accuracy


# Define the flower clients. The Flower clients use a simple CNN adapted from ‘PyTorch: A 60 Minute Blitz’
# (https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)

class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load model and data
net = Net()
trainloader, testloader = load_data()


"""
After loading the data set with load_data() we define the Flower interface.

The Flower server interacts with clients through an interface called Client. When the server selects 
a particular client for training, it sends training instructions over the network. The client receives 
those instructions and calls one of the Client methods to run your code (i.e., to train the neural network 
we defined earlier).

Flower provides a convenience class called NumPyClient which makes it easier to implement the Client 
interface when your workload uses PyTorch. Implementing NumPyClient usually means defining the following
methods (set_parameters is optional though):

1 . get_parameters: returns the model parameters as a NumPy array.

2. set_parameters (optional): update the local model weights with the parameters received from the server. 

3. fit: - set the local model weights
        - train the local model
        - receive the updated local model weights

4. evaluate: test the local model


which can be implemented in the following way:

"""

class CifarClient(fl.client.NumPyClient):
    def get_parameters(self):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(net, trainloader, epochs=1)
        return self.get_parameters(), len(trainloader), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(net, testloader)
        return float(loss), len(testloader), {"accuracy":float(accuracy)}


#  create an instance of our class CifarClient and add one line to actually run this client
fl.client.start_numpy_client("[::]:8080", client=CifarClient())
"""
    That’s it for the client. We only have to implement Client or NumPyClient and call 
    fl.client.start_client() or fl.client.start_numpy_client(). The string "[::]:8080" tells the client 
    which server to connect to. In our case we can run the server and the client on the same machine, 
    therefore we use "[::]:8080". If we run a truly federated workload with the server and clients running 
    on different machines, all that needs to change is the server_address we point the client at.

"""