
# coding: utf-8

# In[ ]:


import matplotlib.pyplot as plt
import os
import time
import math
import random

import utils as util

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import pytorch_ssim


# In[ ]:


if torch.cuda.is_available():
    print("*** " + torch.cuda.get_device_name(torch.cuda.current_device()) + " ***")
else:
    print("WARNING : CUDA NOT AVAILABLE")


# In[ ]:


def crop(im, size, N):
    I = torch.ones(im.size(0)+1-size[0]).multinomial(N,replacement=True)
    J = torch.ones(im.size(1)+1-size[1]).multinomial(N,replacement=True)
    out = torch.FloatTensor(N, size[0],size[1]).zero_()
    for k,(i,j) in enumerate(zip(I,J)):
        out[k] = im[i:i+size[0],j:j+size[1]]
    return(out)

Faces = []
src = '/home/pphan/Downloads/yalefaces/'
for file in os.scandir(src):
    im = torch.from_numpy(plt.imread(file.path))
    Faces.append(crop(im,(232,232),20).div(255))
Faces = F.upsample(torch.cat(Faces).unsqueeze(1), size=128, mode='bilinear').data
print(Faces.size())

Text = []
src = '/home/pphan/Downloads/Projet_Steganographie/generated_secret'
for file in os.scandir(src):
    im = torch.from_numpy(plt.imread(file.path))
    Text.append(im.mean(dim=2))
Text = torch.stack(Text).unsqueeze(1)
print(Text.size())


# In[ ]:


class conv_block(nn.Module):
    def __init__(self, inp_channels, out_channels, down=True):
        super(conv_block, self).__init__()

        self.inp = inp_channels
        self.out = out_channels
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.inp, self.out, 3, padding=1, bias=False),
            nn.LeakyReLU())
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.out, self.out, 3, stride=down+1, padding=1, bias=False),
            nn.BatchNorm2d(self.out), nn.LeakyReLU())

    def forward(self, X):
        c = self.conv1(X)
        return(c, self.conv2(c))


# In[ ]:


class deconv_block(nn.Module):
    def __init__(self, inp_channels, out_channels):
        super(deconv_block, self).__init__()

        self.inp = inp_channels
        self.out = out_channels
        
        self.upconv = nn.Sequential(
            nn.ConvTranspose2d(self.inp, self.out, 2, stride=2, bias=False),
            nn.BatchNorm2d(self.out), nn.ReLU())

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(self.inp, self.out, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.out), nn.ReLU())

        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(self.out, self.out, 3, padding=1, bias=False),
            nn.ReLU())

    def forward(self, X, feature):
        return(self.deconv2(self.deconv1(torch.cat((feature,self.upconv(X)), dim=1))))


# In[ ]:


class contract(nn.Module):
    def __init__(self):
        super(contract, self).__init__()
        
        self.conv1 = conv_block(1,64)
        self.conv2 = conv_block(64,128)
        self.conv3 = conv_block(128,256)
        self.conv4 = conv_block(256,512, down=False)
    
    def forward(self, X):
        f0,X = self.conv1(X) #(64, 128, 128),(64, 64, 64)
        f1,X = self.conv2(X) #(128, 64, 64),(128, 32, 32)
        f2,X = self.conv3(X) #(256, 32, 32),(256, 16, 16)
        _,X = self.conv4(X) #(512, 16, 16)
        return(X, (f0,f1,f2))


# In[ ]:


class expand(nn.Module):
    def __init__(self):
        super(expand, self).__init__()
        
        self.deconv1 = deconv_block(512,256)
        self.deconv2 = deconv_block(256,128)
        self.deconv3 = deconv_block(128,64)
        self.deconv4 = nn.Sequential(nn.Conv2d(64, 1, 1), nn.Tanh())
    
    def forward(self, X, features):
        return(0.5+0.5*self.deconv4(self.deconv3(self.deconv2(self.deconv1(X, features[2]), features[1]), features[0])))


# In[ ]:


class encryptor_decryptor(nn.Module):
    def __init__(self):
        super(encryptor_decryptor, self).__init__()
        
        self.contract_source = contract()
        self.contract_message = contract()
        self.mix = nn.Sequential(nn.Conv2d(1024, 512, 3, padding=1), nn.LeakyReLU())
        self.expand_source = expand()
        
        self.contract_output = contract()
        self.expand_output = expand()
    
    def forward(self, source, message):
        source,features = self.contract_source(source)
        message,_ = self.contract_message(message)
        hidden = self.expand_source(self.mix(torch.cat((source,message), dim=1)), features)
        
        output,features = self.contract_output(hidden)
        return(hidden, self.expand_output(output, features))


# In[ ]:


class Trainer:
    def __init__(self, batch_size=8):
        self.num_layers = 0
        self.num_params = 0
        
        self.model = encryptor_decryptor().cuda()
        for p in self.model.parameters():
            self.num_params += p.view(-1).size(0)
        self.model.apply(self.init_weights)
#         self.model.load_state_dict(torch.load('saved_models/gan/truc.model'))
        print("# layers : {}, # params : {}".format(self.num_layers,self.num_params))
        
        self.optim = torch.optim.Adam(self.model.parameters(), lr=0.0002, betas=(0.5,0.999))
        
        self.batch_size = batch_size
        
        self.epoch = 0
        self.times = []
        self.enc_losses_min = []
        self.enc_losses_max = []
        self.dec_losses_min = []
        self.dec_losses_max = []
        
    def init_weights(self,m):
        G_leaky = nn.init.calculate_gain('leaky_relu')
        G_relu = nn.init.calculate_gain('relu')
        G_tanh = nn.init.calculate_gain('tanh')
        if type(m)==nn.Conv2d:
            if m.out_channels == 1:
                nn.init.xavier_normal(m.weight, gain=G_tanh)
                self.num_layers += 1
            else:
                nn.init.xavier_normal(m.weight, gain=G_leaky)
                self.num_layers += 1
            if m.bias is not None:
                nn.init.constant(m.bias, val=0)
        elif type(m)==nn.ConvTranspose2d:
            nn.init.xavier_normal(m.weight, gain=G_relu)
            self.num_layers += 1
            if m.bias is not None:
                nn.init.constant(m.bias, val=0)
    
    def new_epoch(self):
        self.epoch += 1
        if self.epoch%10==0:
            torch.save(self.model.state_dict(),'saved_models/gan/steganosaurus_e{}.model'.format(self.epoch))
    
    def add_losses(self, losses, t):
        self.times.append(self.epoch + t)
        self.enc_losses_min.append(min(losses[0]))
        self.enc_losses_max.append(max(losses[0]))
        self.dec_losses_min.append(min(losses[1]))
        self.dec_losses_max.append(max(losses[1]))
    
    def disp(self, display):
        if display is not None:
            display[0].clear()
            display[0].plot(self.times,self.enc_losses_min,'r')
            display[0].plot(self.times,self.enc_losses_max,'r')
            display[0].plot(self.times,self.dec_losses_min,'b')
            display[0].plot(self.times,self.dec_losses_max,'b')
            display[1].canvas.draw()
    
    def draw(self, display, data):
        if display is not None:   
            for k,ax in enumerate(display[0]):
                ax.clear()
                ax.imshow(data[0][k].cpu().numpy(), cmap='gray')
            for k,ax in enumerate(display[1]):
                ax.clear()
                ax.imshow(data[1][k].cpu().numpy(), cmap='gray')
            for k,ax in enumerate(display[2]):
                ax.clear()
                ax.imshow(data[2][k].cpu().numpy(), cmap='gray')
            for k,ax in enumerate(display[3]):
                ax.clear()
                ax.imshow(data[3][k].cpu().numpy(), cmap='gray')
            display[4].canvas.draw()

    def train(self, max_epochs=500, points_per_epoch=1, display=None):
        eps = 1e-7
        points = [int(n)-1 for n in torch.linspace(0,Text.size(0)//self.batch_size,1+points_per_epoch)[1:]]
        self.model.train()
        
        ep = 0
        keep = True
        enc_losses,dec_losses = [],[]
        while keep and ep<max_epochs:
            try:
                self.new_epoch()
                ep += 1
                shuffle_f = torch.randperm(Text.size(0))
                shuffle_t = torch.randperm(Text.size(0))
                for mb in range(Text.size(0)//self.batch_size):
                    
                    source = Variable(Faces[shuffle_f[mb*self.batch_size:(mb+1)*self.batch_size]].cuda(), requires_grad=False)
                    message = Variable(Faces[shuffle_t[mb*self.batch_size:(mb+1)*self.batch_size]].cuda(), requires_grad=False)
                    hidden,output = self.model(source, message)
                    
                    lossE = F.mse_loss(hidden,source)
                    lossD = F.mse_loss(output,message)
                    alpha = 0.8
                    loss = alpha*lossE + (1-alpha)*lossD
                    self.optim.zero_grad()
                    loss.backward()
                    self.optim.step()

                    enc_losses.append(lossE.data[0])
                    dec_losses.append(lossD.data[0])
                    
                    if mb in points:
                        self.add_losses((enc_losses,dec_losses),
                                        float(mb/(Text.size(0)//self.batch_size))-1)
                        self.model.eval()
                        source = Variable(Faces[shuffle_f[mb*self.batch_size:mb*self.batch_size+3]].cuda(), requires_grad=False)
                        message = Variable(Faces[shuffle_t[mb*self.batch_size:mb*self.batch_size+3]].cuda(), requires_grad=False)
                        hidden,output = self.model(source, message)
                        self.draw((display[1],display[2],display[3],display[4],display[0]),
                                  (message.data.squeeze(),source.data.squeeze(),
                                   hidden.data.squeeze(),output.data.squeeze(),))
                        self.model.train()
                        
                        self.disp((display[5],display[0]))
                        enc_losses,dec_losses = [],[]
                                
            except KeyboardInterrupt:
                keep = False


# In[ ]:


trainer = Trainer(batch_size=16)

fig = plt.figure(figsize=(10,10))
msg = [fig.add_subplot(4,4,k) for k in [1,5,9]]
src = [fig.add_subplot(4,4,k) for k in [2,6,10]]
hid = [fig.add_subplot(4,4,k) for k in [3,7,11]]
out = [fig.add_subplot(4,4,k) for k in [4,8,12]]
board = fig.add_subplot(4,1,4)
plt.ion()
fig.show()
fig.canvas.draw()


# In[ ]:


trainer.train(points_per_epoch=1, display=(fig,msg,src,hid,out,board))

