
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 00:28:43 2020

@author: tasty_000
"""

#%%
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

#%%

x = torch.randn(100, 1)*10
y = x + 3*torch.randn(100,1)
plt.plot(x.numpy(), y.numpy(), 'o')
plt.ylabel('y')
plt.xlabel('x')
#%%

class LR(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)   
    def forward(self, x):
        pred = self.linear(x)
        return pred
#%%

torch.manual_seed(1)        
model = LR(1,1)
#%%

[w,b] = model.parameters()
def get_params():
    return (w[0][0].item(), b[0].item())
#%%
def plot_fit(title):
    plt.title = title
    
    w1, b1 = get_params()
    x1 = np.array([-30,30])
    y1 = w1*x1 + b1
    plt.plot(x1,y1, 'r')
    plt.scatter(x,y)
    plt.show()
#%%
plot_fit('initial_model')
 #%%   
#specifcy what needs to be optimized + learning rate
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.001)

epochs = 100
losses = []
for i in range(epochs):
    #make predictions
    y_pred = model.forward(x)
    #compare prediction to output
    loss = criterion(y_pred, y)
    print('epoch:', i, "loss:", loss.item())
        
    #optimize weights of model to minimize loss
    losses.append(loss)   
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
plot_fit('trained_model')