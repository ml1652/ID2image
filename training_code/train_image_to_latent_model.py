# -*- coding: utf-8 -*-
"""
Created on Tue May 12 17:44:28 2020

@author: Mingrui
"""

from InterFaceGAN.models.stylegan_generator import StyleGANGenerator
from models.latent_optimizer import PostSynthesisProcessing
from models.image_to_latent import ImageToLatent, ImageLatentDataset
from models.losses import LogCoshLoss
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
from glob import glob
from tqdm import tqdm_notebook as tqdm
#from tqdm.notebook import tqdm
import numpy as np


augments = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

image_size = 256


directory = "./InterFaceGAN/dataset_directory/"
filenames = sorted(glob(directory + "*.jpg"))

train_filenames = filenames[0:48000]
validation_filenames = filenames[48000:]

dlatents = np.load(directory + "wp.npy")

train_dlatents = dlatents[0:48000]
validation_dlatents = dlatents[48000:]

train_dataset = ImageLatentDataset(train_filenames, train_dlatents, transforms=augments)
validation_dataset = ImageLatentDataset(validation_filenames, validation_dlatents, transforms=augments)

train_generator = torch.utils.data.DataLoader(train_dataset, batch_size=32)
validation_generator = torch.utils.data.DataLoader(validation_dataset, batch_size=32)

#Instantiate Model
image_to_latent = ImageToLatent(image_size).cuda()
optimizer = torch.optim.Adam(image_to_latent.parameters())
criterion = LogCoshLoss()

#Train Model
epochs = 20
validation_loss = 0.0

progress_bar = tqdm(range(epochs))
for epoch in progress_bar:    
    running_loss = 0.0
    
    image_to_latent.train()
    for i, (images, latents) in enumerate(train_generator, 1):
        optimizer.zero_grad()

        images, latents = images.cuda(), latents.cuda()
        pred_latents = image_to_latent(images)
        loss = criterion(pred_latents, latents)
        loss.backward()
        
        optimizer.step()
        
        running_loss += loss.item()
        progress_bar.set_description("Step: {0}, Loss: {1:4f}, Validation Loss: {2:4f}".format(i, running_loss / i, validation_loss))
    
    validation_loss = 0.0
    
    image_to_latent.eval()
    for i, (images, latents) in enumerate(validation_generator, 1):
        with torch.no_grad():
            images, latents = images.cuda(), latents.cuda()
            pred_latents = image_to_latent(images)
            loss =  criterion(pred_latents, latents)
            
            validation_loss += loss.item()
    
    validation_loss /= i
    progress_bar.set_description("Step: {0}, Loss: {1:4f}, Validation Loss: {2:4f}".format(i, running_loss / i, validation_loss))

#save moodel 
torch.save(image_to_latent.state_dict(), "C:/Users/Mingrui/Desktop/Github/pytorch_stylegan_encoder/image_to_latent.pt")

#load Model 
image_to_latent = ImageToLatent(image_size).cuda()
image_to_latent.load_state_dict(torch.load("image_to_latent.pt"))
image_to_latent.eval()
