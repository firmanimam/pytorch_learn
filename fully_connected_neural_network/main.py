import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.optim import AdamW

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Hyperparameter
INPUT_SIZE = 784
HIDDEN_SIZE = 500
NUM_CLASSES = 10
NUM_EPOCHS = 3
BATCH_SIZE = 100
LR = 0.001

#Dataset
train_dataset = torchvision.datasets.MNIST(root=r'',
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

model = FCNN(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES).to(device)

#Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-2)

#Train loop

# Number of batches in one epoch
total_step = len(train_loader)
for epoch in range(NUM_EPOCHS):
    for i, (images, labels) in enumerate(train_loader):

        #Flatten each image into a 1D vector
        images = images.reshape(-1, INPUT_SIZE).to(device)

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
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, INPUT_SIZE).to(device)
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


