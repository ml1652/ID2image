# -*- coding: utf-8 -*-
import os
import torch
from glob import glob
from tqdm import tqdm
import numpy as np
import torch.utils.data as Data
import matplotlib.pyplot as plt
from models.latent_optimizer import VGGFaceProcessing
from models.vgg_face2 import resnet50_scratch_dag
from utilities.images import load_images
from tensorboardX import SummaryWriter
from models.regressors import VGGToLatent, VGGLatentDataset

writer = SummaryWriter('vgg_latent_model')

num_trainsets = 47800
Inserver = False

directory = "./dataset_directory/"
filenames = sorted(glob(directory + "*.jpg"))
dlatents = np.load(directory + "WP.npy")
final_dlatents = []
for i in filenames:
    name = int((i.split('\\')[-1]).split('.')[0])
    final_dlatents.append(dlatents[name])
dlatents = np.array(final_dlatents)

train_dlatents = dlatents[0:num_trainsets]
validation_dlatents = dlatents[num_trainsets:]


def generate_vgg_descriptors(filenames):
    vgg_face_dag = resnet50_scratch_dag(
        './Trained_model/resnet50_scratch_dag.pth').cuda().eval()
    descriptors = []
    vgg_processing = VGGFaceProcessing()
    for i in filenames:
        image = load_images([i])
        image = torch.from_numpy(image).cuda()
        image = vgg_processing(image)
        feature = vgg_face_dag(image).cpu().detach().numpy()
        descriptors.append(feature)
    return np.concatenate(descriptors, axis=0)


descriptor_file = directory + 'descriptors.npy'

if os.path.isfile(descriptor_file):
    descriptor = np.load(directory + "descriptors.npy")
else:
    descriptor = generate_vgg_descriptors(filenames)
    np.save(descriptor_file, descriptor)

train_descriptors = descriptor[0:num_trainsets]
validation_descriptors = descriptor[num_trainsets:]

train_dataset = VGGLatentDataset(train_descriptors, train_dlatents)
validation_dataset = VGGLatentDataset(validation_descriptors, validation_dlatents)

train_generator = torch.utils.data.DataLoader(train_dataset, batch_size=32)
validation_generator = torch.utils.data.DataLoader(validation_dataset, batch_size=32)

# Instantiate Model
vgg_to_latent = VGGToLatent().cuda()
optimizer = torch.optim.Adam(vgg_to_latent.parameters(), lr=0.001)


def criterion(feat1, feat2):
    return -torch.nn.functional.cosine_similarity(feat1, feat2).abs().mean()


# Train Model
epochs = 20
validation_loss = 0.0
progress_bar = tqdm(range(epochs))
Loss_list = []
Traning_loss = []
Validation_loss = []
for epoch in progress_bar:
    running_loss = 0.0
    vgg_to_latent.train()

    for i, (vgg_descriptors, latents) in enumerate(train_generator, 1):
        optimizer.zero_grad()
        vgg_descriptors, latents = vgg_descriptors.cuda(), latents.cuda()
        pred_latents = vgg_to_latent(vgg_descriptors)
        loss = criterion(pred_latents, latents)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        progress_bar.set_description(
            "Step: {0}, Loss: {1:4f}, Validation Loss: {2:4f}".format(i, running_loss / i, validation_loss))
    traning_loss = running_loss / i
    Traning_loss.append(traning_loss)
    writer.add_scalar('Test/traning_loss', traning_loss, epoch)
    validation_loss = 0.0
    vgg_to_latent.eval()

    for i, (vgg_descriptors, latents) in enumerate(validation_generator, 1):
        with torch.no_grad():
            vgg_descriptors, latents = vgg_descriptors.cuda(), latents.cuda()
            pred_latents = vgg_to_latent(vgg_descriptors)
            loss = criterion(pred_latents, latents)
            validation_loss += loss.item()

    validation_loss = validation_loss / i
    Validation_loss.append(validation_loss)
    writer.add_scalar('Test/validation_loss', validation_loss, epoch)
    progress_bar.set_description(
        "Step: {0}, Loss: {1:4f}, Validation Loss: {2:4f}".format(i, running_loss / i, validation_loss))

writer.close()

y1 = Traning_loss
y2 = Validation_loss
plt.subplot(2, 1, 1)
plt.plot(y1, 'o-')
plt.title('training loss vs. epoches')
plt.ylabel('training loss')
plt.subplot(2, 1, 2)
plt.plot(y2, '.-')
plt.xlabel('validation loss vs. epoches')
plt.ylabel('validation loss')
plt.savefig("vgg_to_latent_accuracy_loss_stylegancroped_cosinesimiliarty.jpg")

# save model
torch.save(vgg_to_latent.state_dict(), '/Trained_model/vgg_to_latent_wp.pt')
