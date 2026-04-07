import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim import AdamW

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Hyperparameter
NUM_CLASSES = 10
NUM_EPOCHS = 3
BATCH_SIZE = 100
LR = 0.001

#Dataset
train_dataset = torchvision.datasets.MNIST(root='',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)

test_dataset = torchvision.datasets.MNIST(root=r'',
                                          train=False,
                                          transform=transforms.ToTensor())

#Dataloader
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=False)

#Fully connected neural network
class FCNN(nn.Module):
    def __init__(self, INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES):
        super(FCNN, self).__init__()
        self.fc1 = nn.Linear(INPUT_SIZE, HIDDEN_SIZE)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(HIDDEN_SIZE, NUM_CLASSES)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)

        return out

class CNN(nn.Module):
    def __init__(self, NUM_CLASSES):
        super(CNN, self).__init__()

        # First convolution block
        # Input:  (batch_size, 1, 28, 28)
        # Output: (batch_size, 16, 14, 14)
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )


        # Second convolution block
        # Input:  (batch_size, 16, 14, 14)
        # Output: (batch_size, 32, 7, 7)
        self.layer2 = nn.Sequential(
            nn.Conv2d(16,32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )       

         # Fully connected layer
         # Input: 32*7*7 = 1568 features
         # Output: num_classes = 10
        self.fc = nn.Linear(32 * 7 * 7, NUM_CLASSES)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1) #Flatten for FC layer
        out = self.fc(out)

        return out       

model = CNN(NUM_CLASSES).to(device)

#Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-2)

#Train loop

# Number of batches in one epoch
total_step = len(train_loader)
for epoch in range(NUM_EPOCHS):
    for i, (images, labels) in enumerate(train_loader):

        #Flatten each image into a 1D vector
        images = images.to(device)

        labels = labels.to(device)

        #Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        #Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print training progress every 100 batches
        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.3f}'
                  .format(epoch+1, NUM_EPOCHS, i+1, total_step, loss.item()))

#Test loop

#Disable gradient
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        #only keep the predicted class index (dont need the value)
        _,predicted = torch.max(outputs.data, 1)

        #Add the number of samples in this batch to the total count
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy on the test set: {} %'.format(100 * correct / total))

    #Save model
    torch.save(model.state_dict(), 'model.ckpt')


