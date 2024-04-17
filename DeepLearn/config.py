import os

dataset_folder = f'archive'
trainDirectory = os.path.join(dataset_folder, 'train')
testDirectory = os.path.join(dataset_folder, 'test')

train_size = 0.9
val_size = 0.1

batch_size = 16
num_of_epochs = 100
LR = 1e-1