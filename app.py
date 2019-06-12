# coding=utf-8

import sys
import os
import torch
from torch import nn
from torchvision import datasets, transforms
from torch import optim
import torch.onnx as torch_onnx
import json


# Check for CUDA resource
if not torch.cuda.is_available():
    sys.exit('Training requires a GPU')

cuda0 = torch.device('cuda')

# Step 1: Set up target metrics for evaluating training

# Define a target loss metric to aim for
target_loss = 0.26
target_accuracy = 90

# Step 2: Perform training for model
# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])

# Training data
trainset = datasets.MNIST('./MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testset = datasets.MNIST('./MNIST_data/', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# Multi-layer perceptron network
model = nn.Sequential(nn.Linear(784, 64),
                      nn.ReLU(),
                      nn.Linear(64, 32),
                      nn.ReLU(),
                      nn.Linear(32, 10),
                      nn.LogSoftmax(dim=1)).cuda(cuda0)

criterion = nn.NLLLoss()
images, labels = next(iter(trainloader))
images = images.view(images.shape[0], -1).to(cuda0)
labels = labels.to(cuda0)

logps = model(images)
loss = criterion(logps, labels)
optimizer = optim.Adam(model.parameters(), lr=0.01)

epochs = 5
training_loss_metric = 1
for e in range(epochs):
    running_loss = 0
    for images, labels in trainloader:
        images, labels = images.to(cuda0), labels.to(cuda0)
        
        images = images.view(images.shape[0], -1)
        optimizer.zero_grad()

        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    else:
        training_loss_metric = running_loss / len(trainloader)
        print(f'Loss: {training_loss_metric}')

print(f'Training Loss: {training_loss_metric}')

# Step 3: Evaluate the quality of the trained model

correct, total_cnt = 0, 0
test_loss_metric, accuracy_metric = 0.0, 0.0
for batch_idx, (test_images, test_labels) in enumerate(testloader):
    test_images, test_labels = test_images.to(cuda0), test_labels.to(cuda0)
    test_images = test_images.view(test_images.shape[0], -1)
    out = model(test_images)
    loss = criterion(out, test_labels)
    _, predicted = torch.max(out.data, 1)
    total_cnt += test_images.data.size()[0]
    correct += (predicted == test_labels.data).sum()
    # smooth average
    test_loss_metric = test_loss_metric * 0.9 + loss.item() * 0.1

print(f'Testing loss: {test_loss_metric}')

accuracy_metric = float(100.0 * correct / len(testset))
print(f'Accuracy: {accuracy_metric}')


img = test_images[0].view(1, 784)

# Only persist the model if we have passed our desired threshold
if training_loss_metric > target_loss or test_loss_metric > target_loss or accuracy_metric < target_accuracy:
    sys.exit('Training failed to meet threshold')

# Step 4: Persist the trained model in ONNX format in the local file system along with any significant metrics

# Output from the neural network are log probabilities, therefore we need to take exponential for probabilities
ps = torch.exp(logps).to(cuda0)

# Save to ONNX format
dummy_input = torch.Tensor(1, 784).to(cuda0)
torch_onnx.export(model,
                  dummy_input,
                  'model.onnx',
                  verbose=False)

# Write metrics
if not os.path.exists("metrics"):
    os.mkdir("metrics")
with open("metrics/trainingloss.metric", "w+") as f:
    json.dump(training_loss_metric, f)
with open("metrics/testloss.metric", "w+") as f:
    json.dump(test_loss_metric, f)
with open("metrics/accuracy.metric", "w+") as f:
    json.dump(accuracy_metric, f)
