import numpy as np
import cv2
from PIL import Image

def load_images(filenames, transformation = None):
    # Images must all be of same shape.
    images = []
    for filename in filenames:
        raw_image = Image.open(filename)
        if transformation is not None:
            raw_image = transformation(raw_image)
        temp_image = np.asarray(raw_image)
        
        # Adjust channel dimension to work with torch.
        if transformation is None:
            temp_image = np.transpose(temp_image, (2,0,1))
        images.append(temp_image)
        images = np.array(images)

    return np.array(images)

def images_to_video(images, save_path, image_size = 1024):
    size = (image_size, image_size)
    fps = 25
    video = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    for i, image in enumerate(images):
        # Channel, width, height -> width, height, channel, then RGB to BGR
        image = np.transpose(image, (1,2,0))
        image = image[:,:,::-1]
        video.write(image.astype(np.uint8))
        
    video.release() 

def save_image(image, save_path):
    image = np.transpose(image, (1, 2, 0)).astype(np.uint8)
    image = Image.fromarray(image)
    image.save(save_path)

    
