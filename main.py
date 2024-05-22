# Demo: Quantum Image Recognition
#
# Copyright (c) 2021-2024, Dynex Developers
# 
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without modification, are
# permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice, this list of
#    conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright notice, this list
#    of conditions and the following disclaimer in the documentation and/or other
#    materials provided with the distribution.
# 
# 3. Neither the name of the copyright holder nor the names of its contributors may be
#    used to endorse or promote products derived from this software without specific
#    prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL
# THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF
# THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

import math
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from torch.utils.data import random_split
from torch.utils.data import Subset, DataLoader
from torch.nn import Module
from torchvision import transforms
from sklearn.linear_model import LogisticRegression
from torchvision.datasets import MNIST

from HybridQRBM.pytorchdnx import dnx
from HybridQRBM.optimizers import RBMOptimizer
from HybridQRBM.samplers import DynexSampler

import warnings
warnings.filterwarnings('ignore')

# Parameters:
INIT_LR    = 1e-3  # initial loss rate for optimizer
BATCH_SIZE = 10000 # number of images per batch
EPOCHS     = 1     # number of training epochs

# Setting up the Dynex QRBM Module
optimizer = RBMOptimizer(
                learning_rate=0.05,
                momentum=0.9,
                decay_factor=1.00005,
                regularizers=()
            );

sampler = DynexSampler(mainnet=False, 
               num_reads=100000, 
               annealing_time=200,  
               debugging=False, 
               logging=True, 
               num_gibbs_updates=1, 
               minimum_stepsize=0.002);

class QModel(nn.Module):
    def __init__(self, n_hidden, steps_per_epoch, sampler, optimizer):
        super().__init__();
        # Dynex Neuromporphic layer
        self.dnxlayer = dnx(n_hidden, steps_per_epoch, sampler=sampler, optimizer=optimizer); 
        
    def forward(self, x):
        x = self.dnxlayer(x);
        return x

# Load images and transform:
class ThresholdTransform(object):
  def __init__(self, thr_255):
    self.thr = thr_255 / 255.  

  def __call__(self, x):
    return (x > self.thr).to(x.dtype)  

data_transformer = transforms.Compose([
        transforms.ToTensor(),
        ThresholdTransform(thr_255=128) 
    ])

trainData = MNIST(root="data", train=True, download=True, transform=data_transformer)
testData = MNIST(root="data", train=False, download=True, transform=data_transformer) 
print("[INFO] MNIST dataset lodaed")

# Choose classes 0,1 for our demo as we are sampling on testnet:
idx1 = torch.tensor(trainData.targets) == 1;
idx2 = torch.tensor(trainData.targets) == 0;
train_mask = idx1 | idx2;
train_indices = train_mask.nonzero().reshape(-1);
train_subset = Subset(trainData, train_indices);

idx3 = torch.tensor(testData.targets) == 1;
idx4 = torch.tensor(testData.targets) == 0;
test_mask = idx3 | idx4;
test_indices = test_mask.nonzero().reshape(-1);
test_subset = Subset(testData, test_indices);

# initialize the train, validation, and test data loaders
trainDataLoader = DataLoader(train_subset, shuffle=False, batch_size=BATCH_SIZE)
testDataLoader = DataLoader(test_subset, shuffle=False, batch_size=BATCH_SIZE)

# define the model:
steps_per_epoch = len(trainDataLoader.dataset) // BATCH_SIZE
model = QModel(2, steps_per_epoch, sampler, optimizer); 

# Sample trainData on Dynex testnet:
for e in range(1, EPOCHS+1):
    print('EPOCH',e,'of',EPOCHS);
    # set the model in training mode
    model.train()
    # loop over the training set
    for (x, y) in trainDataLoader:
        # send the input to the device
        (x, y) = (x.to('cpu'), y.to('cpu'))
        # perform a forward pass and calculate the training loss
        pred = model(x);
    
print('FOUND MODEL ACCURACY:',np.array(model.dnxlayer.acc).max(),'%')

# Now visualize the result by reconstructing images against testData:
num_samp = 0;
num_batches = 0;
for batch_idx, (inputs, targets) in enumerate(testDataLoader):
    num_samp += len(inputs);
    num_batches += 1;
    
data = [];
data_labels = [];
error = 0;
for i in range(0, 150):
    inp = np.array(inputs[i].flatten().tolist()); 
    tar = np.array(targets[i].tolist())
    data.append(inp)
    data_labels.append(tar)
data = np.array(data)
data_labels = np.array(data_labels)

_, features = model.dnxlayer.sampler.predict(data, num_particles=10,num_gibbs_updates=1)

fig = plt.figure(figsize=(10, 7));
fig.suptitle('Reconstructed Dataset (50 samples)', fontsize=16)
rows = 5;
columns = 10;
for i in range(0,50):
    fig.add_subplot(rows, columns, i+1)
    plt.imshow(features[i].reshape(28,28))
    marker=str(str(data_labels[i]))
    plt.title(marker)
    plt.axis('off');
plt.savefig('result.png')


