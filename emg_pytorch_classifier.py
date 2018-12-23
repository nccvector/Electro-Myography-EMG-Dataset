'''
# This is a Neural Network made with pytorch and it uses all 80 features for classification. 
# You need pytorch and numpy to run it. 
# You can try out different hidden layer sizes, num_epochs, 
# batch_size, learning_rate and test_data_percentage.
# Try running more epochs at a lower learning rate for better accuracy
# Mess with input(features) and classes only if you have got the good understanding of data
# Add more layers to NN if you get an understanding of pytorch
# Read the readme files inside the data folders for more information
'''

import numpy as np
from numpy import genfromtxt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.utils.data as Data
import os
import matplotlib.pyplot as plt

input_size = 80 # number of features
hidden_size1 = 30
hidden_size2 = 30
num_classes = 7 # number of classes
num_epochs = 10 # epochs to run
batch_size = 1000
learning_rate = 0.01
test_data_percentage = 0.2 # 20 percent data for testing/validation

### Preparing Data ###
imported_data = genfromtxt('extracted_features_and_labeled_dataset(easiest to work with)/emg_all_features_labeled.csv', delimiter=',')
samples = len(imported_data[:,0])
# Accending Sort
# imported_data = imported_data[np.argsort(imported_data[:,80])]

# Descending Sort
# imported_data = imported_data[np.argsort(imported_data[:,80])[::-1]]

# Shuffle
np.random.shuffle(imported_data)

data = imported_data[:,0:input_size]
labels = imported_data[:,input_size]

# Scaling features between 0 and 1
for i in range(data.shape[1]):
    data[:,i] = abs(data[:,i])
    max_val = max(data[:,i])
    if max_val >= 1:
        data[:,i] = data[:,i]/max_val

# Converting labels to One-Hot
labels_one_hot = []
for l in labels:
    temp = np.zeros(num_classes)
    temp[int(l-1)] = 1
    labels_one_hot.append(temp)
labels = np.asarray(labels_one_hot, dtype=np.float32)

# Splitting test and train data
test_data = data[0:int(test_data_percentage*samples),:] # test:  first 30 percent
train_data = data[int(test_data_percentage*samples):,:] # train:  remaining data
test_labels = labels[0:int(test_data_percentage*samples),:] # test:  first 30 percent
train_labels = labels[int(test_data_percentage*samples):,:] # train:  remaining data


train_dataset = Data.TensorDataset(
    torch.from_numpy(train_data).float(),
    torch.from_numpy(train_labels).long())

test_dataset = Data.TensorDataset(
    torch.from_numpy(test_data).float(),
    torch.from_numpy(test_labels).long())


# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


# Making a dictionary defining training and validation sets
dataloders = dict()
dataloders['train'] = train_loader
dataloders['val'] = test_loader

dataset_sizes = {'train': int(samples*(1-test_data_percentage)), 'val': int(samples*test_data_percentage)}


# Defining Neural Network Class
class Net(nn.Module):
        def __init__(self, input_size, hidden_size1, hidden_size2, num_classes):
                super(Net, self).__init__()
                self.fc1 = nn.Sequential(
                        nn.Linear(input_size,hidden_size1),
                        nn.ReLU()
                )
                self.fc2 = nn.Sequential(
                        nn.Linear(hidden_size1,hidden_size2),
                        nn.ReLU()
                )
                self.fc3 = nn.Sequential(
                        nn.Linear(hidden_size2,num_classes)
                )
        # Forward prop
        def forward(self, x):
                out = self.fc1(x)
                out = self.fc2(out)
                out = self.fc3(out)
                return out

# Defining Train Function
def train_model(model, criterion, optimizer, num_epochs):
    f = open("Iterations.txt", "w+")
    best_model_wts = model.state_dict()
    best_val_acc = 0.0
    best_train_acc = 0.0
    for epoch in range(num_epochs):
        print('-' * 10)
        print('-' * 10)
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode
            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            for data in dataloders[phase]:
                # get the inputs
                inputs, label = data
                # wrap them in Variable
                if use_gpu:
                    inputs = Variable(inputs.cuda())
                    labels = Variable(label.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(label)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                outputs = model(inputs)
                preds = torch.max(outputs, 1)[1]
                loss = criterion(outputs, torch.max(labels,1)[1])
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                # statistics
                running_loss += loss.data[0]
                for ind in range(len(preds)):
                    if preds[ind] == torch.max(label,1)[1][ind]:
                        running_corrects += 1
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            #Print it out Loss and Accuracy and in the file torchvision
            print('{} Loss: {:.8f} Accuracy: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            f.write('{} Loss: {:.8f} Accuracy: {:.4f}\n'.format(phase, epoch_loss, epoch_acc))
            # deep copy the model
            if phase == 'val' and epoch_acc > best_val_acc:
                best_val_acc = epoch_acc
                best_model_wts = model.state_dict()
            if phase == 'train' and epoch_acc > best_train_acc:
                best_train_acc = epoch_acc
                best_model_wts = model.state_dict()
    f.close()
    print('Best val Acc: {:4f}'.format(best_val_acc))
    model.load_state_dict(best_model_wts)
    return model, best_train_acc, best_val_acc


use_gpu = torch.cuda.is_available()

# Defining net object
net = Net(input_size, hidden_size1, hidden_size2, num_classes)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

if use_gpu:
    model_ft, train_acc, test_acc = train_model(net.cuda(), criterion, optimizer, num_epochs)
else:
    model_ft, train_acc, test_acc = train_model(net, criterion, optimizer, num_epochs)

# Saving the model with best validation accuracy
torch.save(model_ft.state_dict(), 'save.pkl')
