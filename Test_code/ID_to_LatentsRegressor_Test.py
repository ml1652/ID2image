import torch
import numpy as np
import cv2
from utilities.images import load_images
from models.latent_optimizer import VGGFaceProcessing
from models.vgg_face2 import resnet50_scratch_dag
from InterFaceGAN.models.stylegan_generator import StyleGANGenerator
from models.regressors import VGGToLatent
from collections import defaultdict

latent_space_dim = 512
run_device = 'cuda'
vgg_to_latent = VGGToLatent().cuda()
vgg_to_latent.load_state_dict(torch.load("vgg_to_latent_wp.pt"))
vgg_to_latent.eval()
img_directory = ""
image_name = 'test_img.png'
descriptor = np.load(img_directory + "descriptors.npy")
vgg_face_dag = resnet50_scratch_dag('./Trained_model/resnet50_scratch_dag.pth').cuda().eval()
vgg_processing = VGGFaceProcessing()
image = load_images([image_name])
image = torch.from_numpy(image).cuda()
image = vgg_processing(image)
feature = vgg_face_dag(image)
latents_to_be_optimized = vgg_to_latent(feature).cpu().detach().numpy()
kwargs = {'latent_space_type': 'WP'}
model = StyleGANGenerator('stylegan_ffhq')
output_dir = 'vgg_latent_result_image'
latent_codes = model.preprocess(latents_to_be_optimized, **kwargs)
outputs = model.easy_synthesize(latent_codes,
                                **kwargs)
results = defaultdict(list)
for key, val in outputs.items():
    if key == 'image':
        for image in val:
            cv2.imwrite(output_dir + image_name, image[:, :, ::-1])
    else:
        results[key].append(val)
