import argparse
import cv2
import torch
import numpy as np
from tqdm import tqdm
from interfacegan.models.stylegan_generator import StyleGANGenerator
from utilities.images import load_images
from models.vgg_face2 import resnet50_scratch_dag
from utilities.image_processing import crop_image_forVGGFACE2, post_porcessing_outputImg
from utilities.feed_into_regressor import vgg_to_face_regreesor, image_to_Vggencoder, \
    feed_into_Image_to_landmarksRegressor, feed_vggFeatures_into_LandmarkRegressor, \
    feed_vggFeatures_into_histogramRegressor, load_image_to_LandmarkRegressor, load_features_to_LandmarkRegressor
from utilities.image_to_Histogram import image_to_hist


def synthesis_image(dlatents):
    if args.latent_type == 'WP':
        generated_image = synthesizer(dlatents)
    elif args.latent_type == 'Z':
        dlatents = mapping(dlatents)
        dlatents = truncation(dlatents)
        generated_image = synthesizer(dlatents)
    return generated_image


def input_image_into_StyleGAN(latents_to_be_optimized):
    generated_image = synthesis_image(latents_to_be_optimized)
    generated_image = post_porcessing_outputImg(generated_image)
    return generated_image


# Earth mover's distance use to calculate histogram loss
def EMDLoss(output, target):
    # output and target must have size nbatch * nhist * nbins
    # We will compute an EMD for each histogram for each batch element
    loss = torch.zeros(output.shape[0], output.shape[1], output.shape[2] + 1).cuda()  # loss: [batch_size, 3, bins_num]
    for i in range(1, output.shape[2] + 1):
        loss[:, :, i] = output[:, :, i - 1] + loss[:, :, i - 1] - target[:, :, i - 1]  # loss:[32,3,20]
    # Compute the EMD
    loss = loss.abs().sum(dim=2)  # loss: [32,3]
    # Sum over histograms
    loss = loss.sum(dim=1)  # loss: [32]
    # Average over batch
    loss = loss.mean()
    return loss


def inversion_loop(img_dir):
    reference_image = load_images(
        [img_dir]
    )

    reference_image = torch.from_numpy(reference_image).cuda()
    reference_image = reference_image.detach()

    if args.latent_type == 'Z':
        if args.vggface_to_latent == True:
            latents_to_be_optimized = vgg_to_face_regreesor('Z', reference_image, vgg_face2_dag)
        else:
            latents_to_be_optimized = torch.randn(1, 512).cuda().requires_grad_(True)
    elif args.latent_type == 'WP':
        if args.vggface_to_latent == True:
            latents_to_be_optimized = vgg_to_face_regreesor('WP', reference_image, vgg_face2_dag)
        else:
            latents_to_be_optimized = torch.zeros((1, 18, 512)).cuda().requires_grad_(True)

    optimizer = torch.optim.SGD([latents_to_be_optimized], lr=args.learning_rate)
    progress_bar = tqdm(range(args.iterations))
    criterion1 = torch.nn.L1Loss()
    criterion2 = torch.nn.MSELoss()
    count = 0
    X_target = image_to_Vggencoder(reference_image, vgg_face2_dag)
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
        X_target = image_to_Vggencoder(reference_image, vgg_face2_dag)
        L_target = feed_vggFeatures_into_LandmarkRegressor(X_target, features_to_landmarkRegressor)
        L_pred = feed_into_Image_to_landmarksRegressor(croped_image, img_to_landmarkRegressor)
        L_pred = L_pred * (224 / 64)
        X_pred = image_to_Vggencoder(croped_image.cuda(), vgg_face2_dag)

        if args.vgg_to_hist == True:
            hist_pred = image_to_hist(croped_image, bins_num)
            ########################################

        if args.vgg_to_hist == True:
            if step <= IdlossAndLandmarkTraingSteps:
                Id_l = criterion1(X_target, X_pred)
                Landmark_l = criterion2(L_target, L_pred)
                hist_l = EMDLoss(hist_target, hist_pred.unsqueeze(0))
                b = args.weight_landmarkloss
                c = args.weight_histloss
                loss = Id_l + b * Landmark_l + c * hist_l
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
            loss = Id_l + b * Landmark_l
            loss.backward(retain_graph=True)
            optimizer.step()
            Id_l = Id_l.item()
            Landmark_l = Landmark_l.item()

        if args.vgg_to_hist == True:
            progress_bar.set_description(
                "Step: {}, Id_loss: {}, Landmark_loss: {}, hist_loss: {}".format(step, Id_l, Landmark_l, hist_l))
            hist_l_plot.append(hist_l)
        else:
            progress_bar.set_description("Step: {}, Id_loss: {}, Landmark_loss: {}".format(step, Id_l, Landmark_l))
        Landmark_loss_plot.append(Landmark_l)
        Id_loss_plot.append(Id_l)

        # write the final reconstructed image
        if count == int(args.iterations):
            image = croped_image.detach().cpu().numpy()[0]
            image = np.transpose(image, (1, 2, 0))
            image = image[:, :, ::-1]
            str1 = './' + save_file_name + '/'
            str2 = (args.image_path).split('/')[-1]
            last_image_save_path = str1 + str2
            cv2.imwrite(last_image_save_path, np.uint8(image))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find the latent space representation of an input image.")
    parser.add_argument("image_path", help="Filepath of the image to be inversed.")
    parser.add_argument("--learning_rate", default=1, help="Learning rate for SGD.", type=float)
    parser.add_argument("--iterations", default=1000, help="Number of optimizations steps.", type=int)
    parser.add_argument("--model_type", default="stylegan_ffhq", help="The model to use from InterFaceGAN repo.",
                        type=str)
    parser.add_argument("--weight_landmarkloss", default=1, help="the weight of landamrkloss in loss  function",
                        type=float)
    parser.add_argument("--latent_type", default='WP', help="type of StyleGAN's latent.", type=str)
    parser.add_argument("--vggface_to_latent", default=True, help="Use vgg_to_latent model to intialise latent",
                        type=bool)
    parser.add_argument("--vgg_to_hist", default=True, help="Use vgg_to_hist model ", type=bool)
    parser.add_argument("--weight_histloss", default=0.01, help="the weight of vgg_to_histogram in loss  function",
                        type=float)
    parser.add_argument("--save_file_name", default='Non_ID_result', help="the folder used to save results", type=str)
    args, other = parser.parse_known_args()

    IdlossAndLandmarkTraingSteps = args.iterations
    save_file_name = args.save_file_name
    latent_space_dim = 512
    bins_num = 10
    landmarks_num = 68
    img_dir = args.image_path

    # StyleGAN generator
    synthesizer = StyleGANGenerator(args.model_type).model.synthesis
    synthesizer = synthesizer.cuda().eval()
    if args.latent_type == 'Z':
        truncation = StyleGANGenerator("stylegan_ffhq").model.truncation
        mapping = StyleGANGenerator("stylegan_ffhq").model.mapping
        mapping = mapping.cuda().eval()
        truncation = truncation.cuda().eval()

    features_to_landmarkRegressor = load_features_to_LandmarkRegressor(landmarks_num)
    img_to_landmarkRegressor = load_image_to_LandmarkRegressor(landmarks_num)
    vgg_face2_dag = resnet50_scratch_dag('./Trained_model/resnet50_scratch_dag.pth').cuda().eval()

    inversion_loop(img_dir)
