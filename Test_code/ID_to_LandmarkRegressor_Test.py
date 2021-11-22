from models.regressors import LandMarksRegressor
from models.latent_optimizer import VGGFaceProcessing
from models.vgg_face2 import resnet50_scratch_dag
import torch
from utilities.images import load_images
from glob import glob
from draw_face_landmark import draw_face_landmark

image_directory = ""
filenames = glob(image_directory + "*.jpg")
pose_record = []
vgg_processing = VGGFaceProcessing()
vgg_face_dag = resnet50_scratch_dag('./Trained_model/resnet50_scratch_dag.pth').cuda().eval()
Train_Muhammad_regressor = True
output_count = 3
landmarks_num = 68
Train_Muhammad_regressor = True
if Train_Muhammad_regressor == True:
    input_features = 512
else:
    input_features = 2048
landmark_regressor = LandMarksRegressor(landmarks_num, input_features).cuda()
landmark_regressor.load_state_dict(torch.load(
    "./Trained_model/Celeba_Regressor_68_landmarks.pt"))
landmark_regressor.eval()
for filename in filenames:
    image = load_images([filename])
    image_torch = torch.from_numpy(image).cuda()
    image = vgg_processing(image_torch)
    descriptors = vgg_face_dag(image).cuda()
    pred_landmarks = landmark_regressor(descriptors)
    draw_face_landmark(filename, pred_landmarks, image_directory)
    pred_landmarks = str(pred_landmarks)
    pose_record.append(filename + pred_landmarks)
