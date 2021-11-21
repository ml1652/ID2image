from models.regressors import ImageToLandmarks_batch
import torch
from glob import glob
import numpy as np
import cv2
import os
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from Draw_face_landmark import draw_face_landmark


class LoadImages(torch.utils.data.Dataset):
    def __init__(self, filenames, transforms=None):
        self.filenames = filenames
        self.transforms = transforms

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]
        image = self.load_image(filename)
        image = Image.fromarray(np.uint8(image))
        image = self.transforms(image)
        image = image * 255
        image = torch.round(image)
        return image

    def load_image(self, filename):
        image = np.asarray(Image.open(filename))
        return image


image_size = 64
augments = transforms.Compose([
    # transforms.Resize(image_size),
    transforms.ToTensor(),
])

image_directory = ''
filenames = glob(image_directory + "*.jpg")
pose_record = []
output_count = 3
landmarks_num = 68
landmark_regressor = ImageToLandmarks_batch(landmark_num=68).cuda()
landmark_regressor.load_state_dict(torch.load("./Trained_model/Image_to_landmarks_Regressor_batchnorm_lr=0.001.pt"))
landmark_regressor.eval()
images = LoadImages(filenames, transforms=augments)
train_generator = torch.utils.data.DataLoader(images, batch_size=1)
for i, image in enumerate(train_generator):
    image = image.cuda()
    resized_img = F.interpolate(image, 64, mode='bilinear')
    pred_landmarks = landmark_regressor(resized_img)
    pred_landmarks = pred_landmarks * (224 / 64)
    draw_face_landmark(image, pred_landmarks,image_directory)

    pred_landmarks = str(pred_landmarks)
    pose_record.append(filenames[i] + pred_landmarks)
