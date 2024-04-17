from torchvision.transforms import RandomHorizontalFlip
from torchvision.transforms import RandomCrop
from torchvision.transforms import Grayscale
from torchvision.transforms import ToTensor
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import config as cfg
from utils import EarlyStopping
from utils import LRScheduler
from emotionNet import EmotionNet
from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt
from collections import Counter
from datetime import datetime
from torch.optim import SGD
import torch.nn as nn
import pandas as pd
import argparse
import torch
import math

# Create an ArgumentParser object
parser = argparse.ArgumentParser()

# Define command-line arguments
parser.add_argument('-m', '--model', type=str, help='Path to save the trained model')
parser.add_argument('-p', '--plot', type=str, help='Path to save the loss/accuracy plot')

# Parse the command-line arguments
args = vars(parser.parse_args())

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Current training device: {device}")

# Initialize a list of preprocessing steps to apply on each image during
# Training / Validation and Testing
train_transform = transforms.Compose([
    Grayscale(num_output_channels=1),
    RandomHorizontalFlip(),
    RandomCrop((48, 48)),
    ToTensor()
])

test_transform = transforms.Compose([
    Grayscale(num_output_channels=1),
    ToTensor()
])

# Load all the imgs within the specified folder and apply different augmentation
train_data = datasets.ImageFolder(cfg.trainDirectory, transform=train_transform)
test_data = datasets.ImageFolder(cfg.testDirectory, transform=test_transform)

# extract the class labels and the total num of class
classes = train_data.classes
num_of_classes = len(classes)
print(f"[INFO] Class labels: {classes}")

# use train samples to generate train/validation set
num_train_samples = len(train_data)
train_size = math.floor(num_train_samples * cfg.train_size)
val_size = math.ceil(num_train_samples * cfg.val_size)
print(f"[INFO] Train samples: {train_size} ...\t Validation samples: {val_size}...")

train_data, val_data = random_split(train_data, [train_size, val_size])
val_data.dataset.transform = test_transform

train_classes = [label for _, label in train_data]

# count each labels within each classes
class_count = Counter(train_classes)
print(f"[INFO] Total sample: {class_count}")

# compute and determine the weights to the applied on each category
# depending on the number of samples available
class_weight = torch.Tensor([len(train_classes) / c
                            for c in pd.Series(class_count).sort_index().values])

# Initialize a placeholder for each target image, and iterate via the train dataset
# Get the weights for each class and modify the default simple weight to its
# Corresponding class weight already computed
sample_weight = [0] * len(train_data)
for idx, (image, label) in enumerate(train_data):
    weight = class_weight[label]
    sample_weight[idx] = weight

# Define a sampler which randomly sample labels from the train dataset
sampler = WeightedRandomSampler(weights=sample_weight, num_samples=len(train_data), replacement=True)

#load our own dataset and store each sample with their corresponding labels
train_dataloader = DataLoader(train_data, batch_size=cfg.batch_size, sampler=sampler)
val_dataloader = DataLoader(val_data, batch_size=cfg.batch_size)
test_dataloader = DataLoader(test_data, batch_size=cfg.batch_size)

model = EmotionNet(num_of_channels=1, num_of_classes=num_of_classes)
model = model.to(device)

optimizer = SGD(model.parameters(), cfg.LR)
criterion = nn.CrossEntropyLoss()

lr_scheduler = LRScheduler(optimizer=optimizer)
early_stopping = EarlyStopping()

train_steps = len(train_dataloader.dataset) // cfg.batch_size
val_steps = len(val_dataloader.dataset) // cfg.batch_size

history = {
    'train_acc': [],
    'train_loss': [],
    'val_acc': [],
    'val_loss': []
}

# Iterate through the epochs
print(f'[INFO] Training the model...')
start_time = datetime.now()

for epoch in range(0, cfg.num_of_epochs):
    print(f'[INFO] epoch: {epoch + 1}/{cfg.num_of_epochs}')
    model.train()
    total_train_loss = 0
    total_val_loss = 0
    train_correct = 0
    val_correct = 0

    for (data, target) in train_dataloader:
        data, target = data.to(device), target.to(device)
        
        predictions = model(data)
        loss = criterion(predictions, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss
        train_correct += (predictions.argmax(1) == target).type(torch.float).sum().item()

    model.eval()

    with torch.set_grad_enabled(False):
        for (data, target) in val_dataloader:
            data, target = data.to(device), target.to(device)

            predictions = model(data)
            loss = criterion(predictions, target)

            total_val_loss += loss
            val_correct += (predictions.argmax(1) == target).type(torch.float).sum().item()

    avg_train_loss = total_train_loss / train_steps
    avg_val_loss = total_val_loss / val_steps

    train_correct = train_correct / len(train_dataloader.dataset)
    val_correct = val_correct / len(val_dataloader.dataset)

    print(f'train loss: {avg_train_loss:.3f} .. train accuracy: {train_correct:.3f}')
    print(f'val loss: {avg_val_loss:.3f} .. val accuracy: {val_correct:.3f}', end='\n\n')

    history['train_loss'].append(avg_train_loss.cpu().detach().numpy())
    history['train_acc'].append(train_correct)
    history['val_loss'].append(avg_val_loss.cpu().detach().numpy())
    history['val_acc'].append(val_correct)

    validation_loss = avg_val_loss.cpu().detach().numpy()
    lr_scheduler(validation_loss)
    early_stopping(validation_loss)

    if early_stopping.early_stop_enabled:
        break

print(f'[INFO] Total training time: {datetime.now() - start_time}')

if device == 'cuda':
    model = model.to("cpu")
torch.save(model.state_dict(), "EmotionNet_model.pth")

#plot the training loss and accuracy overflow
plt.style.use('ggplot')
plt.figure()
plt.plot(history['train_acc'], label='train_acc')
plt.plot(history['val_acc'], label = 'val_acc')
plt.plot(history['train_loss'], label='train_loss')
plt.plot(history['val_loss'], label='val_loss')
plt.ylabel('Loss/Accuracy')
plt.xlabel('#No of Epochs')
plt.title('Training Loss and Accuracy on FER2013')
plt.legend(loc='upper right')
plt.savefig('EmotionNet_plot2.png')

model.to(device)
with torch.set_grad_enabled(False):
    model.eval()
    predictions = []
    for (data, _) in test_dataloader:
        data = data.to(device)
        output = model(data)
        output = output.argmax(axis=1).cpu().numpy()
        predictions.extend(output)

# Evaluate the Network
print('[INFO] evaluating network...')
actual = [label for _, label in test_data]
print(classification_report(actual, predictions, target_names=test_data.classes))
