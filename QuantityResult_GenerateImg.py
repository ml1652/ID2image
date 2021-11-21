import argparse
import cv2
from tqdm import tqdm
import torch
from InterFaceGAN.models.stylegan_generator import StyleGANGenerator
from models.regressors import VGGToLatent,VGGToHist,LandMarksRegressor
from utilities.images import load_images
from models.latent_optimizer import VGGFaceProcessing
from models.vgg_face2 import resnet50_scratch_dag
from models.vgg_face import vgg_face_dag
from models.regressors import ImageToLandmarks_batch
import torch.nn.functional as F
import numpy as np
import dlib

parser = argparse.ArgumentParser(description="Find the latent space representation of an input image.")
parser.add_argument("image_path", help="Filepath of the image to be encoded.")
parser.add_argument("--learning_rate", default=1, help="Learning rate for SGD.", type= float)
parser.add_argument("--iterations", default=1000, help="Number of optimizations steps.", type=int)
parser.add_argument("--model_type", default="stylegan_ffhq", help="The model to use from InterFaceGAN repo.", type=str)
parser.add_argument("--weight_landmarkloss", default=1 , help="the weight of landamrkloss in loss  function",type= float)
parser.add_argument("--Inserver", default= False , help="Number of optimizations steps.", type=bool)
parser.add_argument("--latent_type", default='WP', help="type of StyleGAN's latent.", type=str)
parser.add_argument("--vggface_to_latent", default= True , help="Use vgg_to_latent model to intialise latent", type=bool)
parser.add_argument("--vgg_to_hist", default= True , help="Use vgg_to_hist model ", type=bool)
parser.add_argument("--weight_histloss", default=0.01, help="the weight of vgg_to_histogram in loss  function",type= float) # deflault = 0.01
parser.add_argument("--save_file_name", default='Non_ID_result', help="the file used to save results",type= str) # deflault = 0.01

args, other = parser.parse_known_args()
IdlossAndLandmarkTraingSteps = args.iterations
lagrangianloss = False
save_file_name = args.save_file_name
latent_space_dim = 512
bins_num = 10
landmarks_num = 68
synthesizer = StyleGANGenerator(args.model_type).model.synthesis
synthesizer = synthesizer.cuda().eval()
landmark_regressor = ImageToLandmarks_batch(landmark_num=68).cuda().eval()
weights_path = './Trained_model/Image_to_landmarks_Regressor_batchnorm_lr=0.001.pt'
landmark_regressor.load_state_dict(torch.load(weights_path))
features_to_landmarkRegressor = LandMarksRegressor(landmarks_num).cuda().eval()
img_dir = args.image_path
vgg_face2_dag = resnet50_scratch_dag('./Trained_model/resnet50_scratch_dag.pth').cuda().eval()
#####input lantent###############################
def crop_image_forVGGFACE2(croped_image):
    target_size = 224
    croped_image = croped_image[:, :, 210:906,164:860]
    croped_image = F.interpolate(croped_image, target_size, mode = 'bilinear')
    return croped_image

def vgg_to_face_regreesor(latent_type, reference_image):
    vgg_to_latent = VGGToLatent().cuda()
    if latent_type == 'WP':
        vgg_to_latent.load_state_dict(torch.load('Trained_model/vgg_to_latent_wp.pt'))
        #vgg_to_latent.load_state_dict(torch.load(r"C:\Users\Mingrui\Desktop\Github\pytorch_stylegan_encoder\Trained_model\vgg_to_latent_stylegancrop_cosinesimiliarty.pt"))

    elif latent_type == 'Z':
        vgg_to_latent.load_state_dict(torch.load("vgg_to_latent_Z.pt"))
    vgg_to_latent.eval()
    vgg_processing = VGGFaceProcessing()
    image = vgg_processing(reference_image)  # vgg16: the vlaue between -150 to 156,dim = [1,3,224,224]
    feature = vgg_face2_dag(image)
    latents_to_be_optimized = vgg_to_latent(feature).detach().cpu().numpy()
    latents_to_be_optimized = torch.from_numpy(latents_to_be_optimized).type(torch.FloatTensor).detach().cuda().requires_grad_(True)
    return latents_to_be_optimized

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
        self.std = torch.tensor([1, 1, 1], device="cuda").view(-1, 1, 1)
        self.mean = torch.tensor([131.0912, 103.8827, 91.4953], device="cuda").view(-1, 1,1)

    def forward(self, image):
        image = image.float()
        if image.shape[2] != 224  or image.shape[3] != 224:
            image = F.adaptive_avg_pool2d(image, self.image_size)

        image = (image - self.mean) / self.std
        return image

def feed_into_Vgg(generated_image):
    features = vgg_face2_dag(generated_image)
    return features

def image_to_Vggencoder(img):
    vgg_processing = VGGFaceProcessing()
    generated_image = vgg_processing(img)  #vgg_processing use PIL load image
    features = feed_into_Vgg(generated_image) #reference iamge[-131 - 163]  gereated_image[-130.09 123.9]
    return features

def feed_into_Image_to_landmarksRegressor(img):
    target_size = 64
    img = F.interpolate(img, target_size, mode='bilinear')
    pred_landmarks = landmark_regressor(img)
    return pred_landmarks
########feed the vgg feathures into landamarksRegresor#####################
if args.Inserver == False:
    features_to_landmarkRegressor.load_state_dict(torch.load(
        "./Trained_model/Celeba_Regressor_68_landmarks.pt"))
else:
    features_to_landmarkRegressor.load_state_dict(torch.load(
        './Trained_model/Celeba_Regressor_68_landmarks.pt'))

def feed_vggFeatures_into_LandmarkRegressor(vgg_features):
    pred_landmarks = features_to_landmarkRegressor(vgg_features)
    return pred_landmarks

def generate_landmark(img, draw_landmark=False):
    if args.Inserver == False:
        Model_PATH = r"C:\Users\Mingrui\Desktop\Github\pytorch_stylegan_encoder\models\shape_predictor_68_face_landmarks.dat"
    else:
        Model_PATH = "./Trained_model/shape_predictor_68_face_landmarks.dat"
    frontalFaceDetector = dlib.get_frontal_face_detector()
    imageRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  #img should be (1024,1024,3) (3,1024,1024)
    faceLandmarkDetector = dlib.shape_predictor(Model_PATH)
    imageRGB = np.uint8(imageRGB).squeeze()
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

####crop the image to feed into VGGface2#################
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

def feed_image_to_vggface(img):
        if args.Inserver == False:
            vgg_face = vgg_face_dag(
                './Trained_model/vgg_face_dag.pth').cuda().eval()
        else:
            vgg_face = vgg_face_dag(
                './Trained_model/vgg_face_dag.pth').cuda().eval()

        features = vgg_face(img)
        return features

#####target image################################
def inversion_loop(img_dir):

    reference_image = load_images(
            [img_dir]
        )

    reference_image = torch.from_numpy(reference_image).cuda()
    reference_image = reference_image.detach()

    if args.latent_type == 'Z':
        if args.vggface_to_latent == True:
            latents_to_be_optimized = vgg_to_face_regreesor('Z', reference_image)
        else:
            latents_to_be_optimized = torch.randn(1, 512).cuda().requires_grad_(True)
    elif args.latent_type == 'WP':
        if args.vggface_to_latent == True:
            latents_to_be_optimized = vgg_to_face_regreesor('WP', reference_image)
        else:
            latents_to_be_optimized = torch.zeros((1, 18, 512)).cuda().requires_grad_(True)
    ##########Training####################################################################################################################
    optimizer = torch.optim.SGD([latents_to_be_optimized], lr=args.learning_rate)

    progress_bar = tqdm(range(args.iterations))
    criterion1 = torch.nn.L1Loss()
    criterion2 = torch.nn.MSELoss()
    image_size = 1024
    count = 0
    X_target = image_to_Vggencoder(reference_image)#[1, 3, 224, 224],dtype=torch.uint8
    if args.vgg_to_hist == True:
        hist_target = feed_vggFeatures_into_histogramRegressor(X_target, bins_num)
        hist_target = hist_target.cuda()
        hist_l_plot = []
    Landmark_loss_plot = []
    Id_loss_plot = []

    for step in progress_bar:
        count += 1
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

        if args.vgg_to_hist == True:
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
        else:
            Id_l = criterion1(X_target, X_pred)
            Landmark_l = criterion2(L_target, L_pred)
            b = args.weight_landmarkloss
            loss = Id_l+ b*Landmark_l
            #latents_to_be_optimized.retain_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            Id_l = Id_l.item()
            Landmark_l = Landmark_l.item()

        if args.vgg_to_hist == True:
            progress_bar.set_description( "Step: {}, Id_loss: {}, Landmark_loss: {}, hist_loss: {}".format(step, Id_l, Landmark_l, hist_l))
            hist_l_plot.append(hist_l)
        else:
            progress_bar.set_description("Step: {}, Id_loss: {}, Landmark_loss: {}".format(step,Id_l,Landmark_l ))
        Landmark_loss_plot.append(Landmark_l)
        Id_loss_plot.append(Id_l)

        # Channel, width, height -> width, height, channel, then RGB to BGR

        if count == int(args.iterations):
            image = generated_image.detach().cpu().numpy()[0]
            image = np.transpose(image, (1, 2, 0))
            image = image[:, :, ::-1]

            str1 = './'+save_file_name+'/'
            str2 = (args.image_path).split('/')[-1]
            last_image_save_path = str1+str2
            cv2.imwrite(last_image_save_path, np.uint8(image))

inversion_loop(img_dir)

