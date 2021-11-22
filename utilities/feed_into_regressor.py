from models.regressors import VGGToLatent, VGGToHist, ImageToLandmarks_batch, LandMarksRegressor
import torch
import torch.nn.functional as F
from utilities.image_processing import VGGFaceProcessing


def vgg_to_face_regreesor(latent_type, reference_image, vgg_face2):
    vgg_to_latent = VGGToLatent().cuda()
    if latent_type == 'WP':
        vgg_to_latent.load_state_dict(torch.load('./Trained_model/vgg_to_latent_wp.pt'))

    elif latent_type == 'Z':
        vgg_to_latent.load_state_dict(torch.load("./Trained_model/vgg_to_latent_Z.pt"))
    vgg_to_latent.eval()
    vgg_processing = VGGFaceProcessing()
    image = vgg_processing(reference_image)
    feature = vgg_face2(image)
    latents_to_be_optimized = vgg_to_latent(feature).detach().cpu().numpy()
    latents_to_be_optimized = torch.from_numpy(latents_to_be_optimized).type(
        torch.FloatTensor).detach().cuda().requires_grad_(True)
    return latents_to_be_optimized


def image_to_Vggencoder(img, vgg_face2):
    vgg_processing = VGGFaceProcessing()
    generated_image = vgg_processing(img)
    features = vgg_face2(generated_image)
    return features


def feed_into_Image_to_landmarksRegressor(img, img_landmark_regressor):
    target_size = 64
    img = F.interpolate(img, target_size, mode='bilinear')
    pred_landmarks = img_landmark_regressor(img)
    return pred_landmarks


def load_features_to_LandmarkRegressor(landmarks_num):
    features_to_landmarkRegressor = LandMarksRegressor(landmarks_num).cuda().eval()
    features_to_landmarkRegressor.load_state_dict(torch.load(
        './Trained_model/Celeba_Regressor_68_landmarks.pt'))
    return features_to_landmarkRegressor


def load_image_to_LandmarkRegressor(landmarks_num):
    img_to_landmarkRegressor = ImageToLandmarks_batch(landmarks_num).cuda().eval()
    weights_path = './Trained_model/Image_to_landmarks_Regressor_batchnorm_lr=0.001.pt'
    img_to_landmarkRegressor.load_state_dict(torch.load(weights_path))
    return img_to_landmarkRegressor


def feed_vggFeatures_into_LandmarkRegressor(vgg_features, features_to_landmarkRegressor):
    pred_landmarks = features_to_landmarkRegressor(vgg_features)
    return pred_landmarks


def feed_vggFeatures_into_histogramRegressor(X_target, bins_num):
    N_HIDDENS = 2
    N_NEURONS = 8
    features_to_hist = VGGToHist(bins_num, BN=True, N_HIDDENS=N_HIDDENS, N_NEURONS=N_NEURONS).cuda().eval()
    hist_file_name = 'vgg_to_hist_bins=' + str(bins_num) + '_HIDDENS=' + str(N_HIDDENS) + '_NEURONS=' + str(
        N_NEURONS) + '.pt'

    features_to_hist.load_state_dict(torch.load('./Trained_model/' + hist_file_name))

    pred_hist = features_to_hist(X_target)
    return pred_hist
