import pandas as pd
from torchvision import transforms
import tarfile
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



def load_data():

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  # unzip tar file on google drive
  tar = tarfile.open("images.tar")
  tar.extractall()
  tar.close()

  # perform transform on dataloaders
  transform = transforms.Compose([transforms.Resize((28, 28)),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

  batch_size = 32

  # read all the data from the extracted folder
  dataset = torchvision.datasets.ImageFolder(root='./Images', transform=transform)

  # calculate length of the data and split it in 80% train and 20% test
  lengths = [int(len(dataset)*0.8), int(len(dataset)*0.2)]
  subsetA, subsetB = torch.utils.data.random_split(dataset, lengths)

  # convert train and test data into dataloaders or batches of data
  trainloader= torch.utils.data.DataLoader(subsetA, batch_size=batch_size, shuffle=True, num_workers=2)

  testloader = torch.utils.data.DataLoader(subsetB, batch_size=batch_size, shuffle=True, num_workers=2)

  return trainloader, testloader


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # Reg No. 20I-2003
        # no of filters in conv1 is 202
        # no of filters in conv2 is 20
        # no of filters in conv3 is 200
        # no of filters in conv4 is 3
        self.conv1 = nn.Conv2d(3, 202, 3, padding = 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(202, 20, 3, padding = 2)
        self.conv3 = nn.Conv2d(20,  200, 3, padding = 2)
        self.conv4 = nn.Conv2d(200, 3, 3, padding = 2)

        # no of neurons in fc1 is 202
        # no of neurons of fc2 is 3
        # output of fc3 is 120
        self.fc1 = nn.Linear(3*3*3, 202)
        self.fc2 = nn.Linear(202, 3)
        self.fc3 = nn.Linear(3, 120)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, 3*3*3)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def loss_opt(net):
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

  return criterion, optimizer

def training_model(trainloader, testloader, criterion, optimizer, net):
  for epoch in range(30):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 500 == 449:    # print every mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 500))
            running_loss = 0.0

  print('Finished Training')


def save_model():
  PATH = './cnn_net.pth'
  torch.save(net.state_dict(), PATH)

def main():

  trainloader, testloader = load_data()
  net = Net()
  criterion, optimizer = loss_opt(net)
  tm = training_model(trainloader, testloader, criterion, optimizer, net)
  sm = save_model()


if __name__ == '__main__':
    main()
