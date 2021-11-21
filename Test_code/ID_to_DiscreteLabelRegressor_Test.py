from models.regressors import CelebaRegressor
from models.latent_optimizer import VGGFaceProcessing
from models.vgg_face2 import resnet50_scratch_dag
import torch
from utilities.images import load_images, images_to_video, save_image
from glob import glob
import numpy as np
import pandas as pd

CeleBaAttribute = ['Wearing_Hat']  # Eyeglasses,Smiling,Mouth_Slightly_Open, Blurry, Wearing_Hat
data = pd.read_csv('./celeba/Anno/list_attr_celeba_name+glass+smiling_hat.csv')
names = data.index
vgg_processing = VGGFaceProcessing()
vgg_face_dag = resnet50_scratch_dag('./Trained_model/resnet50_scratch_dag.pth').cuda().eval()
droped_table = pd.read_csv('./celeba/Anno/imglist_after_crop.csv')
leaved_img_name = droped_table['File_Name']
image_directory = './celeba/data/'
image_path = sorted(glob(image_directory + "*.jpg"))
pose_record = []

for label in CeleBaAttribute:
    celeba_regressor = CelebaRegressor().cuda()
    celeba_regressor.load_state_dict(
        torch.load('./Trained_model/celebaregressor_' + label + '.pt'))
    celeba_regressor.eval()
    Labels_table = pd.read_csv(
        "./celeba/Anno/list_attr_celeba_name+glass+smiling_hat.csv",
        sep=',',
        header=0,
    )
    keys = Labels_table['File_Name'].tolist()
    values = Labels_table[label].tolist()
    celeba_labels = dict(zip(keys, values))

    for i in image_path[-10000:]:
        image = load_images([i])
        image = torch.from_numpy(image).cuda()
        image = vgg_processing(image)

        vgg_descriptors = vgg_face_dag(image).cuda()
        pred_label = celeba_regressor(vgg_descriptors)
        pred_label = pred_label.detach().cpu().numpy()
        pred_discrete_label = np.round(pred_label)

        filename = i.split('\\')[-1]

        real_value = int(celeba_labels[filename])
        if real_value == pred_discrete_label:
            pose_record.append(
                filename + ' ' + 'CorrectPredict' + ' ' + 'Pred Value: ' + str(pred_label) + ' ' + 'Real Value: ' + str(
                    real_value))
        else:
            pose_record.append(
                filename + ' ' + 'WrongPredict' + ' ' + 'Pred Value: ' + str(pred_label) + ' ' + 'Real Value: ' + str(
                    real_value))

    save_path = './Diagram/' + label + '' + 'PredResult.txt'
    with open(save_path, 'w') as f:
        for pose in pose_record:
            f.write(pose + '\n')
