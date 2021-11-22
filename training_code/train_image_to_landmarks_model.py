import cv2
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
from models.regressors import ImageToLandmarks_batch
from torchvision import transforms
from tqdm import tqdm
from PIL import Image

image_directory = "./datasets/CelebA/"
image_size = 64

augments = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor()
])


class ImageLabelDataset(torch.utils.data.Dataset):
    def __init__(self, filenames, labels, transforms=None):
        self.filenames = filenames
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]
        label = self.labels[index]

        image = self.load_image(filename)
        image = Image.fromarray(np.uint8(image))

        image = self.transforms(image)
        image = image * 255
        image = torch.round(image)

        return image, label

    def load_image(self, filename):
        image = np.asarray(Image.open(filename))

        return image


data = pd.read_csv(
    "./datasets/CelebA/tformed_landmark_68point.txt",
    sep=' ',
    header=None,
)
data = data.to_numpy()
names = data[:, 0].tolist()
filenames = [f"{image_directory}/{x}" for x in names]
label_sets = np.stack(data[:, 1:].astype('float64'))
label_sets = label_sets * (64 / 224)
total_dataset_num = len(filenames)
num_validationsets = 40000
num_trainsets = total_dataset_num - num_validationsets
train_filenames = filenames[0:num_trainsets]
validation_filenames = filenames[num_trainsets:]
train_label = label_sets[0:num_trainsets]
validation_label = label_sets[num_trainsets:]
train_dataset = ImageLabelDataset(train_filenames, train_label, transforms=augments)
validation_dataset = ImageLabelDataset(validation_filenames, validation_label, transforms=augments)
train_generator = torch.utils.data.DataLoader(train_dataset, batch_size=16)
validation_generator = torch.utils.data.DataLoader(validation_dataset, batch_size=16)

# Instantiate Model
image_to_latent = ImageToLandmarks_batch(landmark_num=68).cuda()
optimizer = torch.optim.SGD(image_to_latent.parameters(), lr=3e-6, weight_decay=5e-4,
                            momentum=0.9)
criterion = torch.nn.MSELoss()
# Train Model
epochs = 150
validation_loss = 0.0
all_training_loss = []
all_validation_loss = []
Training_loss_epoch_list = []
Validation_loss_epoch_list = []

progress_bar = tqdm(range(epochs))
for epoch in progress_bar:
    running_loss = 0.0
    running_loss_in_epoch = []
    validation_loss_in_epoch = []
    running_count = 0

    image_to_latent.train()
    for i, (images, latents) in enumerate(train_generator, 1):
        try:
            optimizer.zero_grad()
            images, latents = images.cuda(), latents.cuda()
            pred_latents = image_to_latent(images)
            loss = criterion(pred_latents.double(), latents.double())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_loss_in_epoch.append(loss.item())
            running_count += 1
        except Exception:
            cv2.imshow('broken image', images[i])
            cv2.waitKey()

    # plot loss vs epoch
    traning_loss_epoch = running_loss / running_count
    Training_loss_epoch_list.append(traning_loss_epoch)

    validation_loss = 0.0
    validation_loss_count = 0
    validation_loss_sum = 0.0
    image_to_latent.eval()
    for i, (images, latents) in enumerate(validation_generator, 1):
        try:
            with torch.no_grad():
                images, latents = images.cuda(), latents.cuda()
                pred_latents = image_to_latent(images)
                loss = criterion(pred_latents.double(), latents.double())

                validation_loss_sum += loss.item()
                validation_loss_count += 1
                validation_loss_in_epoch.append(loss.item())
                validation_loss += loss.item()
        except Exception:
            cv2.imshow('broken image', images[i])
            cv2.waitKey()

    validation_loss /= i
    # plot loss vs epoch
    validation_loss_epoch = validation_loss_sum / validation_loss_count
    Validation_loss_epoch_list.append(validation_loss_epoch)
    progress_bar.set_description(
        "Step: {0}, Loss: {1:4f}, Validation Loss: {2:4f}".format(epoch, traning_loss_epoch, validation_loss_epoch))

# save moodel
torch.save(image_to_latent.state_dict(),
           "./Trained_model/image_to_landmarks_regressor.pt")


# plot the loss at certain intervals
def every(count):
    i = 0

    def filter_fn(item):
        nonlocal i
        nonlocal count
        result = i % count == 0
        i += 1
        return result

    return filter_fn


def flatten(items):
    result = []
    for item in items:
        result += item
    return result
    # return [item for sublist in items for item in sublist]


# plot loss_epoch
subdiagram = 2
y1 = Training_loss_epoch_list
y2 = Validation_loss_epoch_list
y1 = y1[1:]
y2 = y2[1:]

# plt.subplot
plt.subplot(subdiagram, 1, 1)
plt.plot(y1, 'o-')
plt.title('training loss vs. epoch')
plt.ylabel('training loss')
plt.yscale("log")
plt.subplot(subdiagram, 1, 2)
plt.plot(y2, '.-')
plt.xlabel('validation loss vs. epoch')
plt.ylabel('validation loss')

plt.savefig("./diagram/accuracy_loss_epoch_img_to_landmarks_model.jpg")

plt.show()
