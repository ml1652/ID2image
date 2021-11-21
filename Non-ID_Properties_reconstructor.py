import argparse

import cv2
from tqdm import tqdm
import numpy as np
import torch
from InterFaceGAN.models.stylegan_generator import StyleGANGenerator
from models.latent_optimizer import LatentOptimizer,LatentOptimizerVGGface,LatentOptimizerLandmarkRegressor
from models.regressors import VGGToLatent, VGGLatentDataset,VGGToHist,LandMarksRegressor
from torchvision import transforms
from models.vgg_face_dag import vgg_face_dag

from models.losses import LatentLoss,IdentityLoss
from utilities.hooks import GeneratedImageHook
from utilities.images import load_images, images_to_video, save_image
from utilities.files import validate_path
from models.latent_optimizer import VGGFaceProcessing,LatentOptimizerVGGface_vgg_to_latent
from models.vgg_face2 import resnet50_scratch_dag

from models.image_to_latent import ImageToLandmarks,ImageToLandmarks_batch
from torch.autograd import Variable
import torch.nn.functional as F
import math
#"C:\Users\Mingrui\Desktop\datasets\StyleGANimge_corp\webimage_alignmentTest\alignment_img11.jpg" --learning_rate 1 --weight_landmarkloss 0.0006 --vgg_to_hist 0.01 --dlatent_path Non_ID_Model_dlatents.npy --iterations 3500
from matplotlib.animation import FFMpegWriter

#"C:\Users\Mingrui\Desktop\datasets\StyleGANimge_corp\webimage_alignmentTest\alignment_img01.jpg" --learning_rate 1 --weight_landmarkloss 0.0006 --dlatent_path Non_ID_Model_dlatents.npy --iterations 1500
#"C:\Users\Mingrui\Desktop\datasets\StyleGANimge_corp\webimage_alignmentTest\alignment_img02.jpg" --learning_rate 0.001 --weight_landmarkloss 0.6 --dlatent_path Non_ID_Model_dlatents.npy --iterations 6000
#"C:\Users\Mingrui\Desktop\Github\pytorch_stylegan_encoder\alignment_image_for_test\alignment_img01.jpg"
#"C:\Users\Mingrui\Desktop\Github\HD-CelebA-Cropper\data\aligned\align_size(224,224)_move(0.250,0.000)_face_factor(0.800)_jpg\data\000006.jpg"
#"C:\Users\Mingrui\Desktop\datasets\StyleGANimge_corp\webimage_alignmentTest\alignment_img01.jpg"
parser = argparse.ArgumentParser(description="Find the latent space representation of an input image.")
parser.add_argument("image_path", help="Filepath of the image to be encoded.")
parser.add_argument("--learning_rate", default=1, help="Learning rate for SGD.", type= float)
parser.add_argument("--iterations", default=1000, help="Number of optimizations steps.", type=int)
parser.add_argument("--model_type", default="stylegan_ffhq", help="The model to use from InterFaceGAN repo.", type=str)
parser.add_argument("dlatent_path", help="Filepath to save the dlatent (WP) at.")
parser.add_argument("--weight_landmarkloss", default=1 , help="the weight of landamrkloss in loss  function",type= float)
parser.add_argument("--Inserver", default=False, help="Number of optimizations steps.", type=bool)
parser.add_argument("--latent_type", default='WP', help="type of StyleGAN's latent.", type=str)
parser.add_argument("--vggface_to_latent", default= True, help="Use vgg_to_latent model to intialise latent", type=bool)
parser.add_argument("--vgg_to_hist", default= True , help="Use vgg_to_hist model ", type=bool)
parser.add_argument("--weight_histloss", default=0.005, help="the weight of vgg_to_histogram in loss  function",type= float) # deflault = 0.01
parser.add_argument("--save_file_name", default='Non_ID_result', help="the file used to save results",type= str) # deflault = 0.01

args, other = parser.parse_known_args()

#IdlossAndLandmarkTraingSteps = args.iterations - 1000
IdlossAndLandmarkTraingSteps = args.iterations
lagrangianloss = False

save_file_name = args.save_file_name
if  args.Inserver == False:
    last_image_save_path = './'+save_file_name+'/last_image_Non_ID_reconstructor_lr='+str(args.learning_rate)+'_b='+str(args.weight_landmarkloss)+'_hist'+str(args.weight_histloss)+'_iter='+str(args.iterations)+'.jpg'
    last_image_save_path = './'+save_file_name+'/last_image_Non_ID_reconstructor_lr='+str(args.learning_rate)+'_b='+str(args.weight_landmarkloss)+'_hist'+str(args.weight_histloss)+'_iter='+str(args.iterations)+'.jpg'

else:
    last_image_save_path = './'+save_file_name+'/last_image_Non_ID_reconstructor_lr='+str(args.learning_rate)+'_b='+str(args.weight_landmarkloss)+'_hist'+str(args.weight_histloss)+'_iter='+str(args.iterations)+'.jpg'
    import matplotlib
    matplotlib.use('Agg')

#####target image################################
reference_image = load_images(
        [args.image_path]
    ) #(1, 3, 224, 224) [0-256]

reference_image = torch.from_numpy(reference_image).cuda()
reference_image = reference_image.detach()

if args.Inserver == False:
    vgg_face_dag = resnet50_scratch_dag(
        r'./resnet50_scratch_dag.pth').cuda().eval()
else:
    vgg_face_dag = resnet50_scratch_dag(
       './Trained_model/resnet50_scratch_dag.pth').cuda().eval()
#####input lantent###############################
def crop_image_forVGGFACE2(croped_image):
    target_size = 224
    croped_image = croped_image[:, :, 210:906,164:860]

    croped_image = F.interpolate(croped_image, target_size, mode = 'bilinear')

    return croped_image

def vgg_to_face_regreesor(latent_type, reference_image):
    vgg_to_latent = VGGToLatent().cuda()
    if latent_type == 'WP':
        #vgg_to_latent.load_state_dict(torch.load("vgg_to_latent_WP_styleGAN.pt"))
        #vgg_to_latent.load_state_dict(torch.load("vgg_to_latent_WP.pt"))
        #vgg_to_latent.load_state_dict(torch.load(r"C:\Users\Mingrui\Desktop\vgg_to_latent_WP_cosineloss.pt"))
        #vgg_to_latent.load_state_dict(torch.load(r"C:\Users\Mingrui\Desktop\Github\pytorch_stylegan_encoder\Trained_model\vgg_to_latent_Testoldversion.pt"))
        #vgg_to_latent.load_state_dict(torch.load(r"C:\Users\Mingrui\Desktop\Github\pytorch_stylegan_encoder\Trained_model\vgg_to_latent_WP.pt"))
        vgg_to_latent.load_state_dict(torch.load('Trained_model/vgg_to_latent_stylegancrop.pt'))
        #vgg_to_latent.load_state_dict(torch.load(r"C:\Users\Mingrui\Desktop\Github\pytorch_stylegan_encoder\Trained_model\vgg_to_latent_stylegancrop_cosinesimiliarty.pt"))

    elif latent_type == 'Z':
        vgg_to_latent.load_state_dict(torch.load("vgg_to_latent_Z.pt"))
    vgg_to_latent.eval()
    vgg_processing = VGGFaceProcessing()
    image = vgg_processing(reference_image)  # vgg16: the vlaue between -150 to 156,dim = [1,3,224,224]
    feature = vgg_face_dag(image)
    latents_to_be_optimized = vgg_to_latent(feature).detach().cpu().numpy()
    latents_to_be_optimized = torch.from_numpy(latents_to_be_optimized).type(torch.FloatTensor).detach().cuda().requires_grad_(True)
    return latents_to_be_optimized


latent_space_dim = 512
if args.latent_type == 'Z':
    if args.vggface_to_latent == True:
        latents_to_be_optimized = vgg_to_face_regreesor('Z', reference_image)
    else:
        latents_to_be_optimized = torch.randn(1, 512).cuda().requires_grad_(True)
    #latents_to_be_optimized = torch.zeros(1, 512).cuda().requires_grad_(True)
elif args.latent_type == 'WP':
    if args.vggface_to_latent == True:
        latents_to_be_optimized =  vgg_to_face_regreesor('WP', reference_image)
    else:
        latents_to_be_optimized = torch.zeros((1, 18, 512)).cuda().requires_grad_(True)

synthesizer = StyleGANGenerator(args.model_type).model.synthesis
synthesizer = synthesizer.cuda().eval()

if args.latent_type == 'Z':
    truncation = StyleGANGenerator("stylegan_ffhq").model.truncation
    mapping = StyleGANGenerator("stylegan_ffhq").model.mapping
    mapping = mapping.cuda().eval()
    truncation = truncation.cuda().eval()

#####styleGAN generator##########################
class PostSynthesisProcessing(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.min_value = -1
        self.max_value = 1

    def forward(self, synthesized_image):
        synthesized_image = (synthesized_image - self.min_value) * torch.tensor(255).float() / (self.max_value - self.min_value)
        synthesized_image = torch.clamp(synthesized_image + 0.5, min=0, max=255)

        return synthesized_image  #the value between 0-255, dim = [1,3,1024,1024]

def synthesis_image(dlatents):
    if args.latent_type == 'WP':
        generated_image = synthesizer(dlatents)
    elif args.latent_type == 'Z':
        dlatents = mapping(dlatents)
        dlatents = truncation(dlatents)
        generated_image = synthesizer(dlatents)
    return generated_image

def pose_porcessing_outputImg(img):
    post_synthesis_processing = PostSynthesisProcessing()
    generated_image = post_synthesis_processing(img)
    return generated_image

def input_image_into_StyleGAN(latents_to_be_optimized):
    generated_image = synthesis_image(latents_to_be_optimized)
    generated_image = pose_porcessing_outputImg(generated_image)
    return generated_image

####feed the StyleGAN generated image into vggface2 encoder###################################

class VGGFaceProcessing(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.image_size = 224  # vggface

        #self.mean = torch.tensor([129.186279296875, 104.76238250732422, 93.59396362304688], device="cuda").view(-1, 1,1)
        #self.mean = torch.tensor([0.5066128599877451, 0.41083287257774204, 0.3670351514629289], device="cuda").view(-1, 1, 1)
        self.std = torch.tensor([1, 1, 1], device="cuda").view(-1, 1, 1)

        self.mean = torch.tensor([131.0912, 103.8827, 91.4953], device="cuda").view(-1, 1,1)


    def forward(self, image):
        #image = image / torch.tensor(255).float()
        image = image.float()
        if image.shape[2] != 224  or image.shape[3] != 224:
            image = F.adaptive_avg_pool2d(image, self.image_size)

        image = (image - self.mean) / self.std

        return image

def feed_into_Vgg(generated_image):

    features = vgg_face_dag(generated_image)
    return features # referece [0 7]

def image_to_Vggencoder(img):
    vgg_processing = VGGFaceProcessing()
    generated_image = vgg_processing(img)  #vgg_processing use PIL load image
    features = feed_into_Vgg(generated_image) #reference iamge[-131 - 163]  gereated_image[-130.09 123.9]
    return features

########feed the image into Image_to_landmarks model#####################
# landmark_model_image_size = 64
# style_gan_transformation = transforms.Compose([
#             transforms.ToPILImage(),
#             transforms.Resize(landmark_model_image_size),
#             transforms.ToTensor(),
#             transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                  std=[0.229, 0.224, 0.225])
#         ])



landmark_regressor = ImageToLandmarks_batch(landmark_num=68).cuda().eval()
weights_path = './Trained_model/Image_to_landmarks_Regressor_batchnorm_lr=0.001.pt'
landmark_regressor.load_state_dict(torch.load(weights_path))

def feed_into_Image_to_landmarksRegressor(img):
    # out = []
    # for x_ in img.cpu():
    #     out.append(style_gan_transformation(x_))
    # generated_image = torch.stack(out).cuda()
    target_size = 64
    #img = img/256
    img = F.interpolate(img, target_size, mode='bilinear')
    #generated_image =  torch.nn.functional.normalize(img,dim=2)

    pred_landmarks = landmark_regressor(img)

    return pred_landmarks

####feed the image into vggface2 encoder###################################


########feed the vgg feathures into landamarksRegresor#####################

landmarks_num = 68
features_to_landmarkRegressor = LandMarksRegressor(landmarks_num).cuda().eval()
LandMarksRegressor
if args.Inserver == False:
    features_to_landmarkRegressor.load_state_dict(torch.load(
        "./Trained_model/Celeba_Regressor_68_landmarks.pt"))
else:
    features_to_landmarkRegressor.load_state_dict(torch.load(
        './Trained_model/Celeba_Regressor_68_landmarks.pt'))

def feed_vggFeatures_into_LandmarkRegressor(vgg_features):
    pred_landmarks = features_to_landmarkRegressor(vgg_features)
    #pred_landmarks = str(pred_landmarks)

    return pred_landmarks


###########loss###########################################

class ID_loss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, target, pred):
        loss = torch.nn.functional.l1_loss(target, pred)
        return loss


class Landmark_loss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, target, pred):
        loss = torch.mean(torch.square(target - pred))
        loss = torch.log(loss)
        return loss


class LatentLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_loss = torch.nn.L1Loss()
        self.l2_loss = torch.nn.MSELoss()

    def forward(self, real_features, generated_features, average_dlatents=None, dlatents=None):
        # Take a look at:
        # https://github.com/pbaylies/stylegan-encoder/blob/master/encoder/perceptual_model.py
        # For additional losses and practical scaling factors.

        loss = 0
        # Possible TODO: Add more feature based loss functions to create better optimized latents.

        # Modify scaling factors or disable losses to get best result (Image dependent).

        # VGG16 Feature Loss
        # Absolute vs MSE Loss
        # loss += 1 * self.l1_loss(real_features, generated_features)
        loss += 1 * self.l2_loss(real_features, generated_features)

        # Pixel Loss
        #         loss += 1.5 * self.log_cosh_loss(real_image, generated_image)

        # Dlatent Loss - Forces latents to stay near the space the model uses for faces.
        if average_dlatents is not None and dlatents is not None:
            loss += 1 * 512 * self.l1_loss(average_dlatents, dlatents)

        return loss

#################################################################################
import dlib
from numpy import linalg as LA

def generate_landmark(img, draw_landmark=False):
    # location of the model (path of the model).
    if args.Inserver == False:
        Model_PATH = r"C:\Users\Mingrui\Desktop\Github\pytorch_stylegan_encoder\models\shape_predictor_68_face_landmarks.dat"
    else:
        Model_PATH = "./Trained_model/shape_predictor_68_face_landmarks.dat"

    # now from the dlib we are extracting the method get_frontal_face_detector()
    # and assign that object result to frontalFaceDetector to detect face from the image with
    # the help of the 68_face_landmarks.dat model
    frontalFaceDetector = dlib.get_frontal_face_detector()

    imageRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  #img should be (1024,1024,3) (3,1024,1024)


    # Now the dlip shape_predictor class will take model and with the help of that, it will show
    faceLandmarkDetector = dlib.shape_predictor(Model_PATH)
    imageRGB = np.uint8(imageRGB).squeeze()
    # Now this line will try to detect all faces in an image either 1 or 2 or more faces
    allFaces = frontalFaceDetector(imageRGB, 0)

    # List to store landmarks of all detected faces
    allFacesLandmark = []

    # Below loop we will use to detect all faces one by one and apply landmarks on them

    for k in range(0, max(1, len(allFaces))):
        # dlib rectangle class will detecting face so that landmark can apply inside of that area
        faceRectangleDlib = dlib.rectangle(int(allFaces[k].left()), int(allFaces[k].top()),
                                           int(allFaces[k].right()), int(allFaces[k].bottom()))

        # Now we are running loop on every detected face and putting landmark on that with the help of faceLandmarkDetector
        detectedLandmarks = faceLandmarkDetector(imageRGB, faceRectangleDlib)

        # count number of landmarks we actually detected on image
        # if k == 0:
        #     print("Total number of face landmarks detected ", len(detectedLandmarks.parts()))

        # Svaing the landmark one by one to the output folder
        for point in detectedLandmarks.parts():
            allFacesLandmark.append([point.x, point.y])
            if draw_landmark:
                imageRGB = cv2.circle(imageRGB, (point.x, point.y), 2, (0, 0, 255), 3)
                #imageRGB[point.x,point.y] = [0,0,255]
                cv2.imshow("preview", imageRGB)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    return np.array(allFacesLandmark)


def align_crop_opencv(img,
                      src_landmarks,
                      standard_landmarks,
                      celeba_standard_landmark,
                      src_celeba_landmark,
                      crop_size=512,
                      face_factor=0.8,
                      align_type='similarity',
                      order=3,
                      mode='edge'):
    """Align and crop a face image by landmarks.

    Arguments:
        img                : Face image to be aligned and cropped.
        src_landmarks      : [[x_1, y_1], ..., [x_n, y_n]].
        standard_landmarks : Standard shape, should be normalized.
        crop_size          : Output image size, should be 1. int for (crop_size, crop_size)
                             or 2. (int, int) for (crop_size_h, crop_size_w).
        face_factor        : The factor of face area relative to the output image.
        align_type         : 'similarity' or 'affine'.
        order              : The order of interpolation. The order has to be in the range 0-5:
                                 - 0: INTER_NEAREST
                                 - 1: INTER_LINEAR
                                 - 2: INTER_AREA
                                 - 3: INTER_CUBIC
                                 - 4: INTER_LANCZOS4
                                 - 5: INTER_LANCZOS4
        mode               : One of ['constant', 'edge', 'symmetric', 'reflect', 'wrap'].
                             Points outside the boundaries of the input are filled according
                             to the given mode.
    """
    # set OpenCV

    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    inter = {0: cv2.INTER_NEAREST, 1: cv2.INTER_LINEAR, 2: cv2.INTER_AREA,
             3: cv2.INTER_CUBIC, 4: cv2.INTER_LANCZOS4, 5: cv2.INTER_LANCZOS4}
    border = {'constant': cv2.BORDER_CONSTANT, 'edge': cv2.BORDER_REPLICATE,
              'symmetric': cv2.BORDER_REFLECT, 'reflect': cv2.BORDER_REFLECT101,
              'wrap': cv2.BORDER_WRAP}

    # check
    assert align_type in ['affine', 'similarity'], 'Invalid `align_type`! Allowed: %s!' % ['affine', 'similarity']
    assert order in [0, 1, 2, 3, 4, 5], 'Invalid `order`! Allowed: %s!' % [0, 1, 2, 3, 4, 5]
    assert mode in ['constant', 'edge', 'symmetric', 'reflect', 'wrap'], 'Invalid `mode`! Allowed: %s!' % ['constant',
                                                                                                           'edge',
                                                                                                           'symmetric',
                                                                                                           'reflect',
                                                                                                           'wrap']

    # crop size
    if isinstance(crop_size, (list, tuple)) and len(crop_size) == 2:
        crop_size_h = crop_size[0]
        crop_size_w = crop_size[1]
    elif isinstance(crop_size, int):
        crop_size_h = crop_size_w = crop_size
    else:
        raise Exception(
            'Invalid `crop_size`! `crop_size` should be 1. int for (crop_size, crop_size) or 2. (int, int) for (crop_size_h, crop_size_w)!')

    # estimate transform matrix
    trg_landmarks = standard_landmarks * max(crop_size_h, crop_size_w) * face_factor + np.array(
        [crop_size_w // 2, crop_size_h // 2])

    if align_type == 'affine':
        tform = cv2.estimateAffine2D(trg_landmarks, src_landmarks, ransacReprojThreshold=np.Inf)[0]
    else:
        tform = cv2.estimateAffinePartial2D(trg_landmarks, src_landmarks, ransacReprojThreshold=np.Inf)[0] #tform{2,3}

    # calcaute the scale of tform
    m1 = np.mat('0;0;1')
    m2 = np.mat('1;0;1')
    p1 = tform.dot(m1)
    p2 = tform.dot(m2)
    scale = LA.norm(p2 - p1)  # defualt is Frobenius norm

    # change the translations part of the transformation matrix for downwarding vertically
    tform[1][2] = tform[1][2] + 20 * scale


    # tform = np.round(tform)
    # numpy to tensor
    #tform = torch.tensor(tform).cuda()
    # tform = torch.tensor(tform)

    # tform = torch.tensor(tform, dtype=torch.float)
    #
    # grid = F.affine_grid(tform.unsqueeze(0),(1,3,224,224),align_corners = True)
    #
    # grid = grid.type(torch.FloatTensor).cuda()
    # output = F.grid_sample(img/256, grid,mode="bilinear", padding_mode="border",align_corners=True)

    # warp image by given transform
    output_shape = (crop_size_h, crop_size_w)
    img_crop = cv2.warpAffine(img, tform, output_shape[::-1], flags=cv2.WARP_INVERSE_MAP + inter[order],
                              borderMode=border[mode])

    # #center crop
    # center_crop_size = 224
    # mid_x, mid_y = int(crop_size_w / 2), int(crop_size_h / 2)
    # mid_y = mid_y +16
    # cw2, ch2 = int(center_crop_size / 2), int(center_crop_size / 2)
    # img_crop = img_crop[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]

    # get transformed landmarks
    tformed_landmarks = cv2.transform(np.expand_dims(src_landmarks, axis=0), cv2.invertAffineTransform(tform))[0]
    tformed_celeba_landmarks = cv2.transform(np.expand_dims(src_celeba_landmark, axis=0), cv2.invertAffineTransform(tform))[0]

    return img_crop, tformed_landmarks, tformed_celeba_landmarks


def align_image_fromStylegan_to_vgg(img):
    img_landmark = generate_landmark(img)

    n_landmark = 68
    if args.Inserver == False:
        standard_landmark_file = 'C:/Users/Mingrui/Desktop/Github/HD-CelebA-Cropper/data/standard_landmark_68pts.txt'
        celeba_standard_landmark = np.loadtxt(r"C:\Users\Mingrui\Desktop\celeba\Anno\standard_landmark_celeba.txt",
                                              delimiter=',').reshape(-1, 5, 2)
        celeba_landmark = np.genfromtxt(r"C:\Users\Mingrui\Desktop\celeba\Anno\list_landmarks_celeba.txt",
                                        dtype=np.float,
                                        usecols=range(1, 5 * 2 + 1), skip_header=2).reshape(-1, 5, 2)
    else:
        standard_landmark_file = './Trained_model/standard_landmark_68pts.txt'
        celeba_standard_landmark = np.loadtxt("./Trained_model/standard_landmark_celeba.txt",
                                              delimiter=',').reshape(-1, 5, 2)
        celeba_landmark = np.genfromtxt("./Trained_model/list_landmarks_celeba.txt",dtype=np.float,
                                        usecols=range(1, 5 * 2 + 1), skip_header=2).reshape(-1, 5, 2)

    standard_landmark = np.genfromtxt(standard_landmark_file, dtype=np.float).reshape(n_landmark, 2)
    move_w = 0
    move_h = 0.25
    standard_landmark[:, 0] += move_w
    standard_landmark[:, 1] += move_h



    i = 0

    img_crop, tformed_landmarks, tformed_celeba_landmarks = align_crop_opencv(img,
                                                                              img_landmark,
                                                                              standard_landmark,
                                                                              celeba_standard_landmark,
                                                                              celeba_landmark[i],
                                                                              crop_size=(
                                                                              224, 224),
                                                                              face_factor=0.8,
                                                                              align_type='similarity',
                                                                              order=1,
                                                                              mode='edge')

    return img_crop
#################################################################################

def computeMNELandmark(target_landmark, pred_landmark):
    sum = 0
    count =0
    for target, pred in zip(target_landmark,pred_landmark):
        for i in range(0, len(target), 2):
            pred_x = int(pred[i])
            pred_y = int(pred[i+1])
            target_x = int(target[i])
            target_y = int(target[i+1])

            point_pred = np.array([pred_x,pred_y])
            point_target = np.array([target_x,target_y])
            L2 = np.linalg.norm(point_target-point_pred)
            sum +=L2

            count += 1
            #choice the corner of the eye to compute the inter_ocular_distance
            if count == 37:
                left_cornereye_x = pred_x
                left_cornereye_y = pred_y
            elif count == 46:
                right_cornereye_x = pred_x
                right_cornereye_y = pred_y

    x = left_cornereye_x - right_cornereye_x
    y = left_cornereye_y - right_cornereye_y
    inter_ocular_distance = math.sqrt((x ** 2) + (y ** 2))
    #Mean normalised error
    MNE = sum / (68 * inter_ocular_distance)

    str1 = './'+save_file_name+'/MNELandmark_lr=' + str(
        args.learning_rate) + '_b=' + str(args.weight_landmarkloss) + '_iter=' + str(args.iterations) + '_'
    str2 = ((args.image_path).split('/')[-1]).split('.')[0]
    save_dir = str1 + str2 + '.txt'
    with open(save_dir, 'w') as f:
        f.write(str(MNE))
    return MNE

def drawLandmarkPoint(img, target_landmark, pred_landmark):
    point_size = 1
    target_point_color = (0, 255, 0)  # BGR
    pred_point_color = (0, 0, 255)
    thickness = 4  # 可以为 4
    image_resize = 224

    if args.Inserver == False:
        str1 = './'+save_file_name+'/LandmarkPlot__lr=' + str(
            args.learning_rate) + '_b=' + str(args.weight_landmarkloss)+'_iter='+str(args.iterations) + '_'
        str2 = (args.image_path).split('/')[-1]
    else:
        str1 = './'+save_file_name+'/LandmarkPlot__lr=' + str(
            args.learning_rate) + '_b=' + str(args.weight_landmarkloss)+'_iter='+str(args.iterations) + '_'
        str2 = (args.image_path).split('/')[-1]
    landmarks_image_save_path = str1 + str2

    img = img.detach().cpu().numpy().squeeze().transpose(1, 2, 0)
    # gi1 = np.uint8(gi1)

    image_draw = img
    image_draw = np.uint8(image_draw)
    if image_draw.shape[2] != image_resize:

        image_draw = cv2.resize(image_draw, (image_resize, image_resize), interpolation=cv2.INTER_AREA)

    image_draw = cv2.cvtColor(image_draw, cv2.COLOR_BGR2RGB)

    for landmark in target_landmark:
        for i in range(0, len(landmark), 2):
            point1 = int(landmark[i])
            point2 = int(landmark[i+1])
            point_tuple = (point1, point2)
            image_draw = cv2.circle(image_draw, point_tuple, point_size, target_point_color, thickness)

        for landmark in pred_landmark:
            for i in range(0, len(landmark), 2):
                point1 = int(landmark[i])
                point2 = int(landmark[i + 1])
                point_tuple = (point1, point2)
                image_draw = cv2.circle(image_draw, point_tuple, point_size, pred_point_color, thickness)
    cv2.imwrite(landmarks_image_save_path, image_draw)

    # cv2.imshow("preview", image_draw)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def CompareLandmarkRegreeesorOnTargetImage(img, target_landmark, pred_landmark):
    point_size = 1
    target_point_color = (0, 255, 0)  # BGR
    pred_point_color = (0, 0, 255)
    thickness = 4  # 可以为 4
    image_resize = 224

    if args.Inserver == False:
        str1 = './'+save_file_name+'/TargetImg_LandmarkPlot_lr=' + str(
            args.learning_rate) + '_b=' + str(args.weight_landmarkloss) + '_iter=' + str(args.iterations) + '_'
        str2 = (args.image_path).split('/')[-1]
    else:
        str1 = './'+save_file_name+'/TargetImg_LandmarkPlot_lr=' + str(
            args.learning_rate) + '_b=' + str(args.weight_landmarkloss) + '_iter=' + str(args.iterations) + '_'
        str2 = (args.image_path).split('/')[-1]
    landmarks_image_save_path = str1 + str2

    img = img.detach().cpu().numpy().squeeze().transpose(1, 2, 0)
    # gi1 = np.uint8(gi1)

    image_draw = img
    image_draw = np.uint8(image_draw)
    if image_draw.shape[2] != image_resize:
        image_draw = cv2.resize(image_draw, (image_resize, image_resize), interpolation=cv2.INTER_AREA)

    image_draw = cv2.cvtColor(image_draw, cv2.COLOR_BGR2RGB)

    for landmark in target_landmark:
        for i in range(0, len(landmark), 2):
            point1 = int(landmark[i])
            point2 = int(landmark[i + 1])
            point_tuple = (point1, point2)
            image_draw = cv2.circle(image_draw, point_tuple, point_size, target_point_color, thickness)

        for landmark in pred_landmark:
            for i in range(0, len(landmark), 2):
                point1 = int(landmark[i])
                point2 = int(landmark[i + 1])
                point_tuple = (point1, point2)
                image_draw = cv2.circle(image_draw, point_tuple, point_size, pred_point_color, thickness)
    cv2.imwrite(landmarks_image_save_path, image_draw)

def draw_targetLandmarkPoint(img, target_landmark):
    point_size = 1
    target_point_color = (0, 255, 0)  # BGR
    pred_point_color = (0, 0, 255)
    thickness = 4  # 可以为 0 、4、8
    if args.Inserver == False:
        str1 = './'+save_file_name+'/TargetLandmarkPlot__lr=' + str(
            args.learning_rate) + '_b=' + str(args.weight_landmarkloss)+'_iter='+str(args.iterations)+ '_'
        str2 = (args.image_path).split('/')[-1]
    else:
        str1 = './'+save_file_name+'/TargetLandmarkPlot__lr=' + str(
            args.learning_rate) + '_b=' + str(args.weight_landmarkloss) +'_iter='+str(args.iterations)+ '_'
        str2 = (args.image_path).split('/')[-1]
    landmarks_image_save_path = str1 + str2

    img = img.detach().cpu().numpy().squeeze().transpose(1, 2, 0)
    # gi1 = np.uint8(gi1)

    image_draw = img
    image_draw = np.uint8(image_draw)
    image_resize = 224
    image_draw = cv2.resize(image_draw, (image_resize, image_resize), interpolation=cv2.INTER_AREA)
    image_draw = cv2.cvtColor(image_draw, cv2.COLOR_BGR2RGB)

    for landmark in target_landmark:
        for i in range(0, len(landmark), 2):
            point1 = int(landmark[i])
            point2 = int(landmark[i+1])
            point_tuple = (point1, point2)
            #point_tuple = (round(point1*(1024/224)), round(point2*(1024/224)))
            image_draw = cv2.circle(image_draw, point_tuple, point_size, target_point_color, thickness)
            #image_draw = cv2.putText(image_draw, str(j), point_tuple, cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),3)

    cv2.imwrite(landmarks_image_save_path, image_draw)

def draw_RegressedLandmarkPoint(img, pred_landmark):
    point_size = 1
    target_point_color = (0, 0, 255)  # BGR
    pred_point_color = (0, 0, 255)
    thickness = 4

    str1 = './'+save_file_name+'/RegressedLandmarkPlot__lr=' + str(
        args.learning_rate) + '_b=' + str(args.weight_landmarkloss) +'_iter='+str(args.iterations)+ '_'
    str2 = (args.image_path).split('/')[-1]
    landmarks_image_save_path = str1 + str2

    img = img.detach().cpu().numpy().squeeze().transpose(1, 2, 0)
    # gi1 = np.uint8(gi1)

    image_draw = img
    image_draw = np.uint8(image_draw)
    image_resize = 224
    image_draw = cv2.resize(image_draw, (image_resize, image_resize), interpolation=cv2.INTER_AREA)
    image_draw = cv2.cvtColor(image_draw, cv2.COLOR_BGR2RGB)

    for landmark in pred_landmark:
        for i in range(0, len(landmark), 2):
            point1 = int(landmark[i])
            point2 = int(landmark[i+1])
            point_tuple = (point1, point2)
            #point_tuple = (round(point1*(1024/224)), round(point2*(1024/224)))
            image_draw = cv2.circle(image_draw, point_tuple, point_size, target_point_color, thickness)
            #image_draw = cv2.putText(image_draw, str(j), point_tuple, cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),3)

    cv2.imwrite(landmarks_image_save_path, image_draw)

def draw_GTLandmark(input_img, target_landmark):
    point_size = 1
    target_point_color = (0, 255, 0)  # BGR
    pred_point_color = (0, 0, 255)
    thickness = 4  # 可以为 0 、4、8
    if args.Inserver == False:
        str1 = './'+save_file_name+'/GroundTruthLandmark__lr=' + str(
            args.learning_rate) + '_b=' + str(args.weight_landmarkloss)+'_iter='+str(args.iterations)+ '_'
        str2 = (args.image_path).split('/')[-1]
    else:
        str1 = './'+save_file_name+'/GroundTruthLandmark__lr=' + str(
            args.learning_rate) + '_b=' + str(args.weight_landmarkloss) +'_iter='+str(args.iterations)+ '_'
        str2 = (args.image_path).split('/')[-1]
    landmarks_image_save_path = str1 + str2

    input_img = input_img.detach().cpu().numpy().squeeze().transpose(1, 2, 0)
    L_groundtruth = feed_into_Image_to_landmarksRegressor(input_img)  # first resize input image from 224 to 64
    L_groundtruth = L_groundtruth * (224 / 64)
    # gi1 = np.uint8(gi1)

    image_draw = input_img
    image_draw = np.uint8(image_draw)
    image_resize = 224
    image_draw = cv2.resize(image_draw, (image_resize, image_resize), interpolation=cv2.INTER_AREA)
    image_draw = cv2.cvtColor(image_draw, cv2.COLOR_BGR2RGB)

    for landmark in L_groundtruth:
        for i in range(0, len(landmark), 2):
            point1 = int(landmark[i])
            point2 = int(landmark[i+1])
            point_tuple = (point1, point2)
            #point_tuple = (round(point1*(1024/224)), round(point2*(1024/224)))
            image_draw = cv2.circle(image_draw, point_tuple, point_size, target_point_color, thickness)
            #image_draw = cv2.putText(image_draw, str(j), point_tuple, cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),3)

    cv2.imwrite(landmarks_image_save_path, image_draw)

####crop the image to feed into VGGface2#################
# def crop_image_forVGGFACE2(croped_image):
#     resize_size = 320
#     target_size = 224
#     move_h = 60
#
#     #croped_image = F.adaptive_avg_pool2d(img, resize_size)
#
#     (w, h) = croped_image.size()[2:]
#     target = [680, 680]
#     x_left = int(w / 2 - target[0] / 2)
#     x_right = int(x_left + target[0])
#
#     y_top = int(h / 2 - target[1] / 2 + move_h)
#     y_bottom = int(y_top + target[1])
#
#     #crop to certain rect area
#     croped_image = croped_image[:, :, y_top:y_bottom, x_left:x_right]
#
#     #resize to 224
#     croped_image = F.interpolate(croped_image, target_size, mode = 'bilinear')
#
#     return croped_image



# import matplotlib.pyplot as plt
# croped_image = croped_image.detach().cpu().numpy().squeeze().transpose(1, 2, 0)
# gi1 = np.uint8(croped_image)
# plt.imshow(gi1)
# plt.show()

class Non_ID_reconstructor(torch.nn.Module):
    def __init__(self,synthesizer):
        super().__init__()

        self.synthesizer = synthesizer.cuda().eval()

    def forward(self, latents_to_be_optimized):
        generated_image = synthesis_image(latents_to_be_optimized)
        generated_image = pose_porcessing_outputImg(generated_image)

        #generated_image = input_image_into_StyleGAN(latents_to_be_optimized)  # [1, 3, 1024, 1024] [0-255]

        croped_image = crop_image_forVGGFACE2(generated_image)

        L_pred = feed_into_Image_to_landmarksRegressor(croped_image)  # first resize input image from 224 to 64
        L_pred = L_pred * (224 / 64)

        X_pred = image_to_Vggencoder(croped_image.cuda())


        return L_pred, X_pred,generated_image


import matplotlib.pyplot as plt
#plt.use('Agg')
def show_image(img):
    gi1 = img.detach().cpu().numpy().squeeze().transpose(1, 2, 0)
    plt.imshow(gi1)
    plt.show()

class SoftHistogram(torch.nn.Module):
    def __init__(self, bins, min, max, sigma):
        super(SoftHistogram, self).__init__()
        self.bins = bins
        self.min = min
        self.max = max
        self.sigma = sigma
        self.delta = float(max - min) / float(bins)
        self.centers = float(min) + self.delta * (torch.arange(bins).float() + 0.5)
        self.centers = torch.nn.Parameter(self.centers, requires_grad=False)
        # self.flatten = torch.nn.Flatten()

    def forward(self, x):
        x = torch.unsqueeze(x, 0) - torch.unsqueeze(self.centers, 1)
        x = torch.sigmoid(self.sigma * (x + self.delta / 2)) - torch.sigmoid(self.sigma * (x - self.delta / 2))
        x = x.sum(dim=1)
        return x

def image_to_hist(croped_image):
    r = croped_image[:, 0, :]
    g = croped_image[:, 1, :]
    b = croped_image[:, 2, :]

    softhist = SoftHistogram(bins_num, min=0, max=255, sigma=1.85).cuda() #sigma=0.04
    r = r.flatten()
    g = g.flatten()
    b = b.flatten()

    hist_r = softhist(r)
    hist_g = softhist(g)
    hist_b = softhist(b)
    num_pix = 224 * 224
    hist_r = hist_r / num_pix
    hist_g = hist_g / num_pix
    hist_b = hist_b / num_pix

    hist_pred = torch.stack((hist_r, hist_g, hist_b))
    return hist_pred

def EMDLoss(output,target):
    # output and target must have size nbatch * nhist * nbins
    # We will compute an EMD for each histogram for each batch element
    loss = torch.zeros(output.shape[0],output.shape[1], output.shape[2]+1).cuda() #loss: [batch_size, 3, bins_num]
    for i in range(1,output.shape[2]+1):
        loss[:,:,i] = output[:,:,i-1] + loss[:,:,i-1] - target[:,:,i-1] #loss:[32,3,20]
    # Compute the EMD
    loss = loss.abs().sum(dim=2) # loss: [32,3]
    # Sum over histograms
    loss = loss.sum(dim=1) #loss: [32]
    # Average over batch
    loss = loss.mean()
    return loss

def draw_SoftHistTest(hist_target, hist_pred):

    hist_target =  hist_target.detach().cpu().numpy().squeeze()
    hist_pred =  hist_pred.detach().cpu().numpy().squeeze()

    subdiagram = 3

    plt.subplot(subdiagram, 1, 1)
    plt.title('channel r')
    plt.xlabel('bin_num')
    plt.ylabel('pixel_num')
    plt.plot(hist_target[0], color="green", linestyle='-', label='target_from_HistRegressor')
    plt.plot(hist_pred[0], color="red", linestyle='-', label='Predict')
    plt.legend()

    plt.subplot(subdiagram, 1, 2)
    plt.title('channel g')
    plt.xlabel('bin_num')
    plt.ylabel('pixel_num')
    plt.plot(hist_target[1], color="green", linestyle='-')
    plt.plot(hist_pred[1], color="red", linestyle='-')

    plt.subplot(subdiagram, 1, 3)
    plt.title('channel b')
    plt.xlabel('bin_num')
    plt.ylabel('pixel_num')
    plt.plot(hist_target[2], color="green", linestyle='-')
    plt.plot(hist_pred[2], color="red", linestyle='-')
#
    str1 = './'+save_file_name+'/SoftHistTest__lr=' + str(
        args.learning_rate) + '_b=' + str(args.weight_landmarkloss) + '_iter=' + str(args.iterations) + '_'
    str2 = (args.image_path).split('/')[-1]


    hist_save_path = str1+str2
    plt.savefig(hist_save_path)
    plt.close('all')
    return

def draw_histogram_target_generateImg(reference_img, generated_img):

    r = reference_img[:, 0, :]
    g = reference_img[:, 1, :]
    b = reference_img[:, 2, :]
    hist_r = torch.histc(r.float(), bins_num, min=0, max=255)
    hist_g = torch.histc(g.float(), bins_num, min=0, max=255)
    hist_b = torch.histc(b.float(), bins_num, min=0, max=255)
    num_pix = 224 * 224
    hist_r = hist_r / num_pix
    hist_g = hist_g / num_pix
    hist_b = hist_b / num_pix

    hist_target = torch.stack((hist_r, hist_g, hist_b)).detach().cpu().numpy()

    r = generated_img[:, 0, :]
    g = generated_img[:, 1, :]
    b = generated_img[:, 2, :]
    hist_r = torch.histc(r.float(), bins_num, min=0, max=255)
    hist_g = torch.histc(g.float(), bins_num, min=0, max=255)
    hist_b = torch.histc(b.float(), bins_num, min=0, max=255)
    num_pix = 224 * 224
    hist_r = hist_r / num_pix
    hist_g = hist_g / num_pix
    hist_b = hist_b / num_pix

    hist_pred = torch.stack((hist_r, hist_g, hist_b)).detach().cpu().numpy()

    hist_data = np.stack((hist_target, hist_pred))
    str1 = './'+save_file_name+'/Histogram_histc_TargetAndGeneratedImgCompare__lr=' + str(
        args.learning_rate) + '_b=' + str(args.weight_landmarkloss) + '_iter=' + str(args.iterations) + '_'
    str2 = ((args.image_path).split('/')[-1]).split('.')[0]
    save_dir = str1 + str2 + '.npy'
    np.save(save_dir, hist_data)


    subdiagram = 3


    plt.subplot(subdiagram, 1, 1)
    # plt.title('channel r')
    # plt.xlabel('bin_num')
    # plt.ylabel('pixel_num')
    plt.plot(hist_target[0],color="red", linestyle='-')
    plt.plot(hist_pred[0], color="red",linestyle=':')
    plt.xticks([])  #trun off the xticks
    plt.yticks([])
    #plt.plot(hist_target[0], color="green", linestyle='-', label='targetImg_histc')
    #plt.plot(hist_pred[0], color="red", linestyle='-', label='Predict')
    # plt.legend()

    plt.subplot(subdiagram, 1, 2)
    plt.plot(hist_target[1], color="green",linestyle='-')
    plt.plot(hist_pred[1], color="green",linestyle=':')
    plt.xticks([]) #trun off the xticks
    plt.yticks([])
    # plt.title('channel g')
    # plt.xlabel('bin_num')
    # plt.ylabel('pixel_num')

    values = np.arange(0,10)

    plt.subplot(subdiagram, 1, 3)
    plt.plot(hist_target[2],color="blue", linestyle='-')
    plt.plot(hist_pred[2],color="blue",linestyle=':')

    plt.xticks(values)
    ax = plt.gca()
    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels[0] = '0'
    labels[-1] = '255'
    ax.set_xticklabels(labels)
    plt.yticks([])
    #plt.title('channel b')
    #plt.xlabel('bin_num')
    #plt.ylabel('pixel_num')


    str1 = './'+save_file_name+'/Histogram_histc_TargetAndGeneratedImgCompare__lr=' + str(
        args.learning_rate) + '_b=' + str(args.weight_landmarkloss) + '_iter=' + str(args.iterations) + '_'
    str2 = (args.image_path).split('/')[-1]
    str2 = str2.split('.')[0]


    hist_save_path = str1+str2+'.pdf'
    plt.savefig(hist_save_path)
    plt.close('all')
    return

def plotThreeloss(Landmark_loss_plot, Id_loss_plot, hist_l_plot):
    subdiagram = 3
    plt.subplot(subdiagram, 1, 1)
    plt.title('LandmarkLoss')
    plt.plot(Landmark_loss_plot, color="red", linestyle='-', label='LandmarkLoss')
    plt.legend()

    plt.subplot(subdiagram, 1, 2)
    plt.title('Id_loss_plot')
    plt.plot(Id_loss_plot, color="green", linestyle='-')

    plt.subplot(subdiagram, 1, 3)
    plt.title('hist_l_plot')
    plt.plot(hist_l_plot, color="blue", linestyle='-')

    if args.Inserver == False:
        str1 = './'+save_file_name+'/Threeloss_plot__lr=' + str(
            args.learning_rate) + '_b=' + str(args.weight_landmarkloss) + '_iter=' + str(args.iterations) + '_'
        str2 = (args.image_path).split('/')[-1]
    else:
        str1 = './'+save_file_name+'/Threeloss_plot__lr=' + str(
            args.learning_rate) + '_b=' + str(args.weight_landmarkloss) + '_iter=' + str(args.iterations) + '_'
        str2 = (args.image_path).split('/')[-1]

    hist_save_path = str1 + str2
    plt.savefig(hist_save_path)
    plt.close('all')

##########Training####################################################################################################################


if lagrangianloss == False:
    optimizer = torch.optim.SGD([latents_to_be_optimized], lr=args.learning_rate)
    #optimizer = torch.optim.Adam([latents_to_be_optimized], lr=args.learning_rate)

progress_bar = tqdm(range(args.iterations))

criterion1 = torch.nn.L1Loss()
# def criterion1(feat1, feat2):
#     # maximize average magnitude of cosine similarity
#     return -torch.nn.functional.cosine_similarity(feat1, feat2).abs().mean()

#criterion1 = torch.nn.CosineSimilarity()
#criterion1 = LatentLoss()
criterion2 = torch.nn.MSELoss()
#criterion2 = Landmark_loss()
if args.vgg_to_hist == True:
    bins_num = 10
    #criterion3 = torch.nn.MSELoss()

image_size = 1024
size = (image_size, image_size)
fps = 25
# if args.Inserver == False:
#     video = cv2.VideoWriter(r'C:\Users\Mingrui\Desktop\Github\pytorch_stylegan_encoder\'+save_file_name+'\vdieo__lr='+str(args.learning_rate)+'_b='+str(args.weight_landmarkloss)+'_'+str(args.image_path[-10:])+'_Non_IDReconstruction.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
# else:
#     video = cv2.VideoWriter(
#         '/scratch/staff/ml1652/StyleGAN_Reconstuction_server/'+save_file_name+'/vdieo__lr='+str(args.learning_rate)+'_b='+str(args.weight_landmarkloss)+'_'+str(args.image_path[-10:])+'_Non_IDReconstruction.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
if args.Inserver == False:
    str1 = './'+save_file_name+'/video__lr=' + str(
        args.learning_rate) + '_b=' + str(args.weight_landmarkloss)+'_iter='+str(args.iterations) + '_'
    str2 = ((args.image_path).split('/')[-1]).split('.')[0]+'.avi'
    video_path = str1 + str2
    video = cv2.VideoWriter( video_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
count = 0



# reference_image = reference_image.detach().cpu().numpy()
# reference_image = reference_image.squeeze()
# reference_image = reference_image.transpose(1, 2, 0)
# reference_image = align_image_fromStylegan_to_vgg(reference_image)
# reference_image = reference_image.transpose(2, 0, 1)
# reference_image = reference_image[np.newaxis, :]
# reference_image = torch.tensor(reference_image) # [1,3,224,224]
# reference_image = reference_image.cuda()
X_target = image_to_Vggencoder(reference_image)#[1, 3, 224, 224],dtype=torch.uint8
L_target = feed_vggFeatures_into_LandmarkRegressor(X_target)

def feed_vggFeatures_into_histogramRegressor(X_target,bins_num):
    N_HIDDENS = 2
    N_NEURONS = 8
    features_to_hist = VGGToHist(bins_num,BN = True, N_HIDDENS = N_HIDDENS,N_NEURONS = N_NEURONS).cuda().eval()
    hist_file_name = 'vgg_to_hist_bins=' + str(bins_num)+'_HIDDENS='+str(N_HIDDENS)+'_NEURONS='+str(N_NEURONS)+'.pt'
    if args.Inserver == False:
        features_to_hist.load_state_dict(torch.load(
            './Trained_model/' + hist_file_name))
    else:
        features_to_hist.load_state_dict(torch.load(
            './Trained_model/'+ hist_file_name))

    pred_hist = features_to_hist(X_target)
    return pred_hist

if args.vgg_to_hist == True:
    hist_target = feed_vggFeatures_into_histogramRegressor(X_target, bins_num)
    hist_target = hist_target.cuda()
    hist_l_plot = []
    #hist_target = torch.histc(reference_image, bin = 60, min=0, max=0)
# for param in Non_ID_reconstructormodel.parameters():
#         param.requires_grad_(False)
loss_plot = []
Landmark_loss_plot = []
Id_loss_plot = []

#latents_to_be_optimized_cpoy = Variable(latents_to_be_optimized.data.clone(), requires_grad=True).cuda()
if lagrangianloss == True:
    lambdA_lists = []
    epsilon  = 0.1 #5
    lambdA = torch.tensor([0.]).cuda().requires_grad_(True)
# str1 = "/scratch/staff/ml1652/StyleGAN_Reconstuction_server/croped_img/"

for step in progress_bar:
    count += 1
    if lagrangianloss == False:
        optimizer.zero_grad()
    generated_image = input_image_into_StyleGAN(latents_to_be_optimized)
    croped_image = crop_image_forVGGFACE2(generated_image)
    X_target = image_to_Vggencoder(reference_image)
    L_target = feed_vggFeatures_into_LandmarkRegressor(X_target)
    L_pred = feed_into_Image_to_landmarksRegressor(croped_image)  # first resize input image from 224 to 64
    L_pred = L_pred * (224 / 64)
    X_pred = image_to_Vggencoder(croped_image.cuda())

    if args.vgg_to_hist == True:

        hist_pred  = image_to_hist(croped_image)
        ########################################
        #hist_l = criterion3(hist_target,hist_pred) # hist_perd = hist_target:[1,3,bin_num]

    if lagrangianloss == True:
        Landmark_l_stopgradient = Landmark_l.detach()
        damp = 10*(epsilon - Landmark_l_stopgradient)
        lagrangian = Id_l - (lambdA-damp) * (epsilon - Landmark_l)

    if count == 1:

        if args.Inserver == False:
            str1 = './'+save_file_name+'/frist_image_Non_ID_reconstructor_lr='+str(args.learning_rate)+'_b='+str(args.weight_landmarkloss)+'_iter='+str(args.iterations)+'_'
            str2 = (args.image_path).split('/')[-1]
        else:
            str1 = './'+save_file_name+'/frist_image_Non_ID_reconstructor_lr='+str(args.learning_rate)+'_b='+str(args.weight_landmarkloss)+'_iter='+str(args.iterations)+'_'
            str2 = (args.image_path).split('/')[-1]
        first_image_save_path = str1+str2
        frist_image = generated_image.detach().cpu().numpy().squeeze().transpose(1, 2, 0)
        first_image = cv2.cvtColor(frist_image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(first_image_save_path, np.uint8(first_image))
        draw_RegressedLandmarkPoint(reference_image, L_pred)


    if lagrangianloss == True:
        lambdA.retain_grad()
        latents_to_be_optimized.retain_grad()
        lagrangian.backward()
        with torch.no_grad():
            latents_to_be_optimized = latents_to_be_optimized - args.learning_rate * latents_to_be_optimized.grad
            lambdA = lambdA + lambdA.grad
            if lambdA < 0:
                lambdA = 0 * lambdA
        latents_to_be_optimized.requires_grad = True
        lambdA.requires_grad = True
    else:
        if args.vgg_to_hist == True:
            # if step <= IdlossAndLandmarkTraingSteps:
            #     Id_l = criterion1(X_target, X_pred)
            #     Landmark_l = criterion2(L_target, L_pred)
            #     b = args.weight_landmarkloss
            #
            #     loss = Id_l +  b * Landmark_l
            # else:
            #     hist_l = EMDLoss(hist_target, hist_pred.unsqueeze(0))  # hist_perd = hist_target:[1,3,bin_num]
            #     c = args.weight_histloss
            #     loss = c*hist_l
            ########train 3 loss together###########################
            # Id_l = criterion1(X_target, X_pred)
            # Landmark_l = criterion2(L_target, L_pred)
            # hist_l = EMDLoss(hist_target, hist_pred.unsqueeze(0))
            # b = args.weight_landmarkloss
            # c = args.weight_histloss
            # loss = Id_l + b * Landmark_l + c*hist_l
            #########################################################
            if step <= IdlossAndLandmarkTraingSteps:
                Id_l = criterion1(X_target, X_pred)
                Landmark_l = criterion2(L_target, L_pred)
                hist_l = EMDLoss(hist_target, hist_pred.unsqueeze(0))
                b = args.weight_landmarkloss
                c = args.weight_histloss
                loss = Id_l + b * Landmark_l + c*hist_l
                Id_l = Id_l.item()
                Landmark_l = Landmark_l.item()
                hist_l = hist_l.item()
            else:
                Id_l = criterion1(X_target, X_pred)
                Landmark_l = criterion2(L_target, L_pred)
                b = args.weight_landmarkloss
                loss = Id_l + b * Landmark_l
                Id_l = Id_l.item()
                Landmark_l = Landmark_l.item()

            loss.backward(retain_graph=True)
            optimizer.step()
            loss = loss.item()
        else:
            Id_l = criterion1(X_target, X_pred)
            Landmark_l = criterion2(L_target, L_pred)
            b = args.weight_landmarkloss
            loss = Id_l+ b*Landmark_l
            #latents_to_be_optimized.retain_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            loss = loss.item()
            Id_l = Id_l.item()
            Landmark_l = Landmark_l.item()

    # if step <= IdlossAndLandmarkTraingSteps:
    #     Landmark_l =  Landmark_l.item()
    #     Id_l = Id_l.item()
    # else:
    #     if args.vgg_to_hist == True:
    #         hist_l = hist_l.item()

    # Id_l = Id_l.item()
    # Landmark_l = Landmark_l.item()
    # hist_l = hist_l.item()

    if count == int(args.iterations):
        if args.Inserver == False:
            str1 = './'+save_file_name+'/last_image_Non_ID_reconstructor_lr='+str(args.learning_rate)+'_b='+str(args.weight_landmarkloss)+'_hist'+str(args.weight_histloss)+'_iter='+str(args.iterations)+'_'
            str2 = (args.image_path).split('/')[-1]
        else:
            str1 = './'+save_file_name+'/last_image_Non_ID_reconstructor_lr='+str(args.learning_rate)+'_b='+str(args.weight_landmarkloss)+'_hist'+str(args.weight_histloss)+'_iter='+str(args.iterations)+'_'
            str2 = (args.image_path).split('/')[-1]
        last_image_save_path = str1+str2
        cv2.imwrite(last_image_save_path, np.uint8(image))
        drawLandmarkPoint(croped_image, L_target, L_pred)
        draw_targetLandmarkPoint(reference_image, L_target)
        CompareLandmarkRegreeesorOnTargetImage(reference_image,L_target, L_pred )

    if args.vgg_to_hist == True:
        progress_bar.set_description( "Step: {}, Id_loss: {}, Landmark_loss: {}, hist_loss: {}".format(step, Id_l, Landmark_l, hist_l))
        # if step <= IdlossAndLandmarkTraingSteps:
        #     progress_bar.set_description("Step: {}, Id_loss: {}, Landmark_loss: {}".format(step, Id_l, Landmark_l))
        # else:
        #     progress_bar.set_description("Step: {}, Id_loss: {}, Landmark_loss: {}, hist_loss: {}".format(step, Id_l, Landmark_l,hist_l))
        hist_l_plot.append(hist_l)



    else:
        progress_bar.set_description("Step: {}, Id_loss: {}, Landmark_loss: {}".format(step,Id_l,Landmark_l ))
    #loss_plot.append(loss)
    Landmark_loss_plot.append(Landmark_l)
    Id_loss_plot.append(Id_l)
    # drawLandmarkPoint(croped_image,L_target, L_pred)
    # drawLandmarkPoint(reference_image,L_target, L_pred)

    # Channel, width, height -> width, height, channel, then RGB to BGR
    image = generated_image.detach().cpu().numpy()[0]
    #image = image*256
    image = np.transpose(image, (1, 2, 0))
    image = image[:, :, ::-1]

    if args.Inserver == False:
        video.write(np.uint8(image))

    if count == int(args.iterations):
        #str1 = "/scratch/staff/ml1652/StyleGAN_Reconstuction_server/croped_img/"
        if args.Inserver == False:
            str1 = './'+save_file_name+'/last_image_Non_ID_reconstructor_lr='+str(args.learning_rate)+'_b='+str(args.weight_landmarkloss)+'_hist'+str(args.weight_histloss)+'_iter='+str(args.iterations)+'_'
            str2 = (args.image_path).split('/')[-1]
        else:
            str1 = './'+save_file_name+'/last_image_Non_ID_reconstructor_lr='+str(args.learning_rate)+'_b='+str(args.weight_landmarkloss)+'_hist'+str(args.weight_histloss)+'_iter='+str(args.iterations)+'_'
            str2 = (args.image_path).split('/')[-1]
        last_image_save_path = str1+str2
        cv2.imwrite(last_image_save_path, np.uint8(image))
        drawLandmarkPoint(croped_image, L_target, L_pred)
        draw_targetLandmarkPoint(reference_image, L_target)
        draw_SoftHistTest(hist_target, hist_pred)
        draw_histogram_target_generateImg(reference_image, croped_image)
        plotThreeloss(Landmark_loss_plot, Id_loss_plot, hist_l_plot)

        CompareLandmarkRegreeesorOnTargetImage(reference_image, L_target, L_pred)

        #computeMNELandmark(L_target, L_pred)

optimized_dlatents = latents_to_be_optimized.detach().cpu().numpy()
np.save(args.dlatent_path, optimized_dlatents)

if args.Inserver == False:
    video.release()

subdiagram = 2
plt.subplot(subdiagram, 1, 1)
plt.plot(Landmark_loss_plot, color="red", linestyle = '-', label = 'loss')
plt.title('Landmark_loss_plot')
plt.subplot(subdiagram, 1, 2)
plt.plot(Id_loss_plot, color="blue", linestyle = '-', label = 'Id_L')
plt.title('Id_loss')
# plt.subplot(subdiagram, 1, 3)
# plt.plot(Landmark_loss_plot, color="green", linestyle = '-', label = 'landmark_L')
# plt.title('landmark_loss')

import matplotlib.pyplot as plt
import os
import numpy as np
import time
from math import *
#
if args.Inserver == False:
    metadata = dict(title='loss_trajectories', artist='Matplotlib',comment='loss_trace')
    ffmpegpath = os.path.abspath("C:/ffmpeg/bin/ffmpeg.exe")
    plt.rcParams["animation.ffmpeg_path"] = ffmpegpath
    writer = FFMpegWriter(fps=5, metadata=metadata)

figure = plt.figure()
# plt.ion() #open interactive mode


def traverse_imgs(writer, Id_loss_plot, Landmark_loss_plot):
    for i in range(args.iterations):
        plt.clf()  # empty plot
        #landmark_ls = np.array(Landmark_loss_plot[i].detach().cpu().numpy())
        landmark_ls = np.array(Landmark_loss_plot[i])
        #ID_ls = np.array(Id_loss_plot[i].detach().cpu().numpy())
        ID_ls = np.array(Id_loss_plot[i])
        landmark_list.append(landmark_ls)
        ID_list.append(ID_ls)
        ax = plt.axes(xlim=(0, 350), ylim=(-1, 2))
        plt.xlabel('landmark_loss 222')
        plt.ylabel('ID_loss')

        plt.plot(landmark_list, ID_list, '-r', markersize=4)
        # plt.draw()
        # plt.savefig(r"C:\Users\Mingrui\AppData\Local\Temp\\" + str(i) + ".png")
        # plt.show()
        writer.grab_frame()
        #plt.pause(0.01)
        plt.close('all')

# if args.Inserver == False:
#     if args.Inserver == False:
#         str1 = './'+save_file_name+'/loss_trajectories_lr=' + str(
#             args.learning_rate) + '_b=' + str(args.weight_landmarkloss) + '_iter=' + str(args.iterations) + '_'
#         str2 = ((args.image_path).split('\\')[-1]).split('.')[0] + '.mp4'
#     else:
#         str1 = '/scratch/staff/ml1652/StyleGAN_Reconstuction_server/learningrate_Test/loss_trajectories_lr=' + str(
#             args.learning_rate) + '_b=' + str(args.weight_landmarkloss) + '_iter=' + str(args.iterations) + '_'
#         str2 = ((args.image_path).split('/')[-1]).split('.')[0] + '.mp4'
#     loss_trajectories_save_path = str1+str2
#     with writer.saving(figure,loss_trajectories_save_path,100):
#         landmark_list = []
#         ID_list = []
#         traverse_imgs(writer, Id_loss_plot, Landmark_loss_plot)

    if args.Inserver == False:
        str1 = './'+save_file_name+'/plotloss__lr=' + str(
            args.learning_rate) + '_b=' + str(args.weight_landmarkloss)+'_iter='+str(args.iterations)+'_'
        str2 = (args.image_path).split('/')[-1]
    else:
        str1 = '/scratch/staff/ml1652/pytorch_stylegan_encoder/'+save_file_name+'/plotloss__lr=' + str(
            args.learning_rate) + '_b=' + str(args.weight_landmarkloss)+'_iter='+str(args.iterations) + '_'
        str2 = (args.image_path).split('/')[-1]
    plot_path = str1+str2
    plt.savefig(plot_path)
    plt.close('all')
if  args.vgg_to_hist == True:
    if args.Inserver == False:
        str1 = './'+save_file_name+'/histogramloss__lr=' + str(
            args.learning_rate) + '_b=' + str(args.weight_landmarkloss)+'_iter='+str(args.iterations)+'_'
        str2 = (args.image_path).split('/')[-1]
    else:
        str1 = '/scratch/staff/ml1652/pytorch_stylegan_encoder/'+save_file_name+'/histogramloss__lr=' + str(
            args.learning_rate) + '_b=' + str(args.weight_landmarkloss)+'_iter='+str(args.iterations) + '_'
        str2 = (args.image_path).split('/')[-1]
    plot_path = str1+str2
    plt.figure()
    plt.title('histogram_loss_plot')
    plt.plot(hist_l_plot, color="red", linestyle='-', )
    plt.savefig(plot_path)
    plt.close('all')

# plt.plot(lambdA_lists, color="red", linestyle = '-', label = 'lambda')
# plt.title('lambda')
# plot_path = 'C:/Users/Mingrui/Desktop/Github/pytorch_stylegan_encoder/'+save_file_name+'/lambda.jpg'
# plt.savefig(plot_path)
#
# lambdA = lambdA.item()
# print('lambda= ' +str(lambdA) )

# for step in progress_bar:
#     count += 1
#
#     optimizer.zero_grad()
#
#     # image_np = generated_image.detach().cpu().numpy()
#     # image_np = image_np.squeeze()
#     # image_np = image_np.transpose(1, 2, 0)
#     # image_aligned = align_image_fromStylegan_to_vgg(image_np) #(224, 224, 3)
#     # image_aligned = np.round(image_aligned)
#     # #imge_to_tensor
#     # image_fed_inLandmarkregressor = image_aligned.transpose(2, 0, 1)
#     # image_fed_inLandmarkregressor = image_fed_inLandmarkregressor[np.newaxis, :]
#     #
#     #
#     # image_fed_inLandmarkregressor = np.uint8(image_fed_inLandmarkregressor)
#     # image_fed_inLandmarkregressor = torch.tensor(image_fed_inLandmarkregressor) # ([1, 3, 224, 224])
#     generated_image = input_image_into_StyleGAN(latents_to_be_optimized)  # [1, 3, 1024, 1024] [0-255]
#
#     ###################################################################
#     tform = torch.tensor([
#         [3.11314244e+00, -1.35652176e-02, 1.64518438e+02],
#          [1.35652176e-02, 3.11314244e+00, 2.11537025e+02]
# ], dtype=torch.float)
#
#     ###################################################################
#     # image_np = generated_image.detach().cpu().numpy()
#     # image_np = image_np.squeeze()
#     # image_np = image_np.transpose(1, 2, 0)
#     # image_aligned = align_image_fromStylegan_to_vgg(image_np) #(224, 224, 3)
#     croped_image = crop_image_forVGGFACE2(generated_image) #input image: [1,3,1024,1024] output: [1,3,224,224]
#
# ######################
#
#
#     L_pred = feed_into_Image_to_landmarksRegressor(croped_image) #first resize input image from 224 to 64
#     L_pred = L_pred * (224 / 64)
#
#     X_pred = image_to_Vggencoder(croped_image.cuda())
#
#     Id_loss = -criterion1(X_target.double(), X_pred.double())
#     Id_loss.register_hook(print)
#     Landmark_loss = criterion2(L_target.double(), L_pred.double())
#     b = args.weight_landmarkloss
#     #loss = b*Id_loss+ Landmark_loss
#     #loss = Id_loss + lambdA*Landmark_loss
#     loss = Id_loss
#     loss.backward(retain_graph=True)
# #################################################################
#
#     loss = loss.item()
#     Id_loss = Id_loss.item()
#     Landmark_loss = Landmark_loss.item()
#     optimizer.step()
#
#     progress_bar.set_description("Step: {}, Loss: {}, Id_loss: {}, Landmark_loss: {}".format(step, loss,Id_loss,Landmark_loss ))
#
#     loss_plot.append(loss)
#     Landmark_loss_plot.append(Landmark_loss)
#     Id_loss_plot.append(Id_loss)
#     # drawLandmarkPoint(croped_image,L_target, L_pred)
#     # drawLandmarkPoint(reference_image,L_target, L_pred)
#
#     # Channel, width, height -> width, height, channel, then RGB to BGR
#     image = generated_image.detach().cpu().numpy()[0]
#     #image = image*256
#     image = np.transpose(image, (1, 2, 0))
#     image = image[:, :, ::-1]
#
#     video.write(np.uint8(image))
#
#     optimized_dlatents = latents_to_be_optimized.detach().cpu().numpy()
#     np.save(args.dlatent_path, optimized_dlatents)
#
#
#
#     if count == int(args.iterations):
#         #str1 = "/scratch/staff/ml1652/StyleGAN_Reconstuction_server/croped_img/"
#
#         if args.Inserver == False:
#             str1 = 'C:/Users/Mingrui/Desktop/Github/pytorch_stylegan_encoder/'+save_file_name+'/last_image_Non_ID_reconstructor_lr='+str(args.learning_rate)+'_b='+str(args.weight_landmarkloss)+'_iter='+str(args.iterations)+'_'
#             str2 = (args.image_path).split('\\')[-1]
#         else:
#             str1 = '/scratch/staff/ml1652/StyleGAN_Reconstuction_server/learningrate_Test/last_image_Non_ID_reconstructor_lr='+str(args.learning_rate)+'_b='+str(args.weight_landmarkloss)+'_iter='+str(args.iterations)+'_'
#             str2 = (args.image_path).split('/')[-1]
#         last_image_save_path = str1+str2
#         cv2.imwrite(last_image_save_path, np.uint8(image))
#         drawLandmarkPoint(croped_image, L_target, L_pred)u
#         draw_targetLandmarkPoint(reference_image, L_target)
#
# video.release()
# subdiagram = 3
# plt.subplot(subdiagram, 1, 1)
# plt.plot(loss_plot, color="red", linestyle = '-', label = 'loss')
# plt.title('total loss')
# plt.subplot(subdiagram, 1, 2)
# plt.plot(Id_loss_plot, color="blue", linestyle = '-', label = 'Id_L')
# plt.title('Id_loss')
# plt.subplot(subdiagram, 1, 3)
# plt.plot(Landmark_loss_plot, color="green", linestyle = '-', label = 'landmark_L')
# plt.title('landmark_loss')
#
# if args.Inserver == False:
#     str1 = 'C:/Users/Mingrui/Desktop/Github/pytorch_stylegan_encoder/Non_ID_result/plotloss__lr=' + str(
#         args.learning_rate) + '_b=' + str(args.weight_landmarkloss)+'_iter='+str(args.iterations)+'_'
#     str2 = (args.image_path).split('\\')[-1]
# else:
#     str1 = '/scratch/staff/ml1652/StyleGAN_Reconstuction_server/learningrate_Test/plotloss__lr=' + str(
#         args.learning_rate) + '_b=' + str(args.weight_landmarkloss)+'_iter='+str(args.iterations) + '_'
#     str2 = (args.image_path).split('/')[-1]
# plot_path = str1+str2
# plt.savefig(plot_path)


#https://blog.csdn.net/Eddy_Wu23/article/details/108797023
#https://discuss.pytorch.org/t/affine-transformation-matrix-paramters-conversion/19522/13
#https://github.com/wuneng/WarpAffine2GridSample

# def show_tensorImg(img):
#     import matplotlib.pyplot as plt
#     gi1 = img.detach().cpu().numpy().squeeze().transpose(1, 2, 0)
#     gi1 = np.uint8(gi1)
#     plt.imshow(gi1)
#     plt.show()
#
#     class SoftHistogram(torch.nn.Module):
#         def __init__(self, bins, min, max, sigma):
#             super(SoftHistogram, self).__init__()
#             self.bins = bins
#             self.min = min
#             self.max = max
#             self.sigma = sigma
#             self.delta = float(max - min) / float(bins)
#             self.centers = float(min) + self.delta * (torch.arange(bins).float() + 0.5)
#
#         def forward(self, x):
#             x = torch.unsqueeze(x, 0) - torch.unsqueeze(self.centers, 1)
#             x = torch.sigmoid(self.sigma * (x + self.delta / 2)) - torch.sigmoid(self.sigma * (x - self.delta / 2))
#             x = x.sum(dim=1)
#             return x

#softhist = SoftHistogram(bins=60, min=0, max=255, sigma=3*25)
# def plot_hist(hist_r_soft, hist_g_soft, hist_b_soft, hist_r, hist_g, hist_b):
#     hist_r_soft = hist_r_soft.detach().cpu().numpy()
#     hist_r = hist_r.detach().cpu().numpy()
#     plt.plot(hist_r_soft, color="red", linestyle='-', )
#     plt.plot(hist_r, color="green", linestyle='-', )
#     plt.show()