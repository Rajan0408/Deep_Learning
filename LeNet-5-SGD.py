#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd  #for SGD
import torch.optim as optim        #optimization

from torch.utils.data import Dataset

from torchvision import datasets,transforms
from torchvision.transforms import ToTensor,Lambda


# In[2]:


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        
        self.conv1 = nn.Conv2d(1,6,5)
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.pool2 = nn.MaxPool2d(2,2)
        self.f1 = nn.Linear(400,120)
        self.f2 = nn.Linear(120,84)
        self.f3 = nn.Linear(84,10)
    
    def forward(self,x):
        
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 400)
        x = F.relu(self.f1(x))
        x = F.relu(self.f2(x))
        x = self.f3(x)
        
        return(F.log_softmax(x, dim =1))


# In[3]:


net1 = LeNet()
net2 = LeNet()
net3 = LeNet()

print(net1,net2,net3)

use_gpu = torch.cuda.is_available()

if use_gpu:
    print("GPU is availabe")
    net1 = net1.cuda()
    net2 = net2.cuda()
    net3 = net3.cuda()


# In[12]:


apply_transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor()])
BatchSize = 2
BatchSize1 = 2

trainset = datasets.MNIST(root = 'data', download  = False, train = True, transform = apply_transform)
trainloader = torch.utils.data.DataLoader(trainset, shuffle=True, num_workers=4, batch_size = BatchSize)

testset = datasets.MNIST(root = 'data', download = False, train = False, transform = apply_transform)
testloader = torch.utils.data.DataLoader(testset, shuffle=True, num_workers = 4, batch_size = BatchSize)

trainset2 = datasets.MNIST(root = 'data', download  = False, train = True, transform = apply_transform)
trainloader2 = torch.utils.data.DataLoader(trainset, shuffle=True, num_workers=4, batch_size = BatchSize)

testset2 = datasets.MNIST(root = 'data', download = False, train = False, transform = apply_transform)
testloader2 = torch.utils.data.DataLoader(testset, shuffle=True, num_workers = 4, batch_size = BatchSize)


# In[13]:


print(trainset,testset,trainset2,testset2)


# In[14]:


criterion = nn.CrossEntropyLoss() 
learning_rate1 = 0.1
optimizer1 = optim.SGD(net1.parameters(), lr=learning_rate1, momentum=0.9) # SGD 
num_epochs = 25

train_loss = []
train_acc = []
for epoch in range(num_epochs):
    
    running_loss = 0.0 
    running_corr = 0
        
    for i,data in enumerate(trainloader):
        inputs,labels = data
        if use_gpu:
            inputs, labels = inputs.cuda(),labels.cuda() 
        # Initializing model gradients to zero
        optimizer1.zero_grad() 
        # Data feed-forward through the network
        outputs1 = net1(inputs)
        # Predicted class is the one with maximum probability
        preds1 = torch.argmax(outputs1,dim=1)
        # Finding the loss
        loss = criterion(outputs1, labels)
        # Accumulating the loss for each batch
        running_loss += loss 
        # Accumulate number of correct predictions
        running_corr += torch.sum(preds1==labels)    
        
    totalLoss1 = running_loss/(i+1)
    # Calculating gradients
    totalLoss1.backward()
    # Updating the model parameters
    # Updating the model parameters
    optimizer1.step()
        
    epoch_loss = running_loss.item()/(i+1)   #Total loss for one epoch
    epoch_acc = running_corr.item()/60000
    
    
         
    train_loss.append(epoch_loss) #Saving the loss over epochs for plotting the graph
    train_acc.append(epoch_acc) #Saving the accuracy over epochs for plotting the graph
       
        
    print('Epoch {:.0f}/{:.0f} : Training loss: {:.4f} | Training Accuracy: {:.4f}'.format(epoch+1,num_epochs,epoch_loss,epoch_acc*100)) 


# In[ ]:





# In[ ]:




