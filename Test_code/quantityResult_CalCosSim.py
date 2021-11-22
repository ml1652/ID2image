import matplotlib.pyplot as plt
import numpy as np
import os
import SSIM
import torch
from tqdm import tqdm
from utilities.images import load_images
from models.latent_optimizer import VGGFaceProcessing
from models.vgg_face import vgg_face_dag
from glob import glob


def feed_image_to_vggface(img):
    vgg_face = vgg_face_dag(
        './Trained_model/vgg_face_dag.pth').cuda().eval()

    features = vgg_face(img)
    return features


class PSNR:
    def __init__(self):
        self.name = "PSNR"

    @staticmethod
    def __call__(img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        return 20 * torch.log10(255.0 / torch.sqrt(mse))


def Calmetrics(GeneImg_dir, RefImg_dir, type, log=False):
    filenames = sorted(glob(GeneImg_dir + "*.jpg"))
    imag_names = [i.split("\\")[-1] for i in filenames]
    Cos_loss = torch.nn.CosineSimilarity()
    SSIM_loss = SSIM.SSIM()
    PSNR_loss = PSNR()
    MSE_loss = torch.nn.MSELoss()
    Cos_output_list = []
    SSIM_output_list = []
    PSNR_output_list = []
    MSE_features_output_list = []
    MSE_img_output_list = []

    def img_processp_intoVgg(img):
        vgg_processing = VGGFaceProcessing()
        img = vgg_processing(img)
        return img

    progress_bar = tqdm(imag_names)
    for img_dir in progress_bar:
        ref_img_256 = load_images([RefImg_dir + img_dir])
        ref_img_256 = torch.from_numpy(ref_img_256).cuda()
        ref_img_256 = ref_img_256.detach()
        ref_img = img_processp_intoVgg(ref_img_256)

        Gen_img_256 = load_images([GeneImg_dir + img_dir])
        Gen_img_256 = torch.from_numpy(Gen_img_256).cuda()
        Gen_img_256 = Gen_img_256.detach()
        Gen_img = img_processp_intoVgg(Gen_img_256)

        ref_img_ex = feed_image_to_vggface(ref_img)
        Gen_img_ex = feed_image_to_vggface(Gen_img)
        Cos_output = Cos_loss(ref_img_ex, Gen_img_ex)
        SSIM_output = SSIM_loss(ref_img_256.float(), Gen_img_256.float())
        PSNR_output = PSNR_loss(ref_img, Gen_img)
        MSE_features_output = MSE_loss(ref_img, Gen_img)
        MSE_img_output = MSE_loss(ref_img_ex, Gen_img_ex)
        Cos_output = Cos_output.detach().cpu().numpy()
        SSIM_output = SSIM_output.detach().cpu().numpy()
        PSNR_output = PSNR_output.detach().cpu().numpy()
        MSE_features_output = MSE_features_output.detach().cpu().numpy()
        MSE_img_output = MSE_img_output.detach().cpu().numpy()
        if log == True:
            Cos_output = np.log(1 + Cos_output)
            SSIM_output = np.log(1 + SSIM_output)
            PSNR_output = np.log(1 + PSNR_output)
        Cos_output_list.append(Cos_output)
        SSIM_output_list.append(SSIM_output)
        PSNR_output_list.append(PSNR_output)
        MSE_features_output_list.append(MSE_features_output)
        MSE_img_output_list.append(MSE_img_output)
        progress_bar.set_description("Finish: {}".format(img_dir))

    save_dir = "./result/MOFA_Test_result/"
    np.save(save_dir + type + "_Cos_output.npy", Cos_output_list)
    np.save(save_dir + type + "_SSIM_output.npy", SSIM_output_list)
    np.save(save_dir + type + "_PSNR_output.npy", PSNR_output_list)
    np.save(save_dir + type + "_MSE_img_output.npy", MSE_img_output_list)
    np.save(save_dir + type + "_MSE_features_output.npy", MSE_features_output_list)
    np.savetxt(save_dir + type + "_Cos_output.csv", Cos_output_list)
    np.savetxt(save_dir + type + "_SSIM_output.csv", SSIM_output_list)
    np.savetxt(save_dir + type + "_PSNR_output.csv", PSNR_output_list)
    np.savetxt(save_dir + type + "_MSE_img_output.csv", MSE_img_output_list)
    np.savetxt(save_dir + type + "_MSE_features_output.csv", MSE_features_output_list)

    Cos_avg = np.average(Cos_output_list)
    SSIM_avg = np.average(SSIM_output_list)
    PSNR_avg = np.average(PSNR_output_list)
    MSE_img_avg = np.average(MSE_img_output_list)
    MSE_features_avg = np.average(MSE_features_output_list)

    with open(save_dir + type + "avgresult.txt", 'w') as f:
        f.write(
            "Average of CosSimilarity: %s\nAverage of SSIM: %s\nAverage of PSNR: %s\nAverage of MSE_img: %s\nAverage of MSE_features: %s" % (
                str(Cos_avg), str(SSIM_avg), str(PSNR_avg), str(MSE_img_avg), str(MSE_features_avg)))


def plot_histgram(npy_dir):
    val = np.load(npy_dir)
    plt.rcParams['font.size'] = 10
    (n, bins, patches) = plt.hist(val, bins=15, color='dodgerblue', histtype='stepfilled', alpha=0.75)
    mid_point = []
    for i in range(len(bins) - 1):
        mid_point.append((bins[i] + bins[i + 1]) / 2)
    plt.plot(mid_point, n, color="red", linestyle='-', label='Cos Similarity')
    plt.title('VGG-Face cosine similarity between oringinal image and inversed image')
    save_dir = os.path.split(npy_dir)[0] + "/MOFA_CosSim_hist.jpg"
    plt.yticks([])
    plt.savefig(save_dir)


type = "ID+LAND+HIST"
GeneImg_dir = "./MOFA_Test_result/MOFA_Test_result_" + type + "/data/"
RefImg_dir = "./MOFA_Test_result/MoFA-test_aligned/data/"
Calmetrics(GeneImg_dir, RefImg_dir, type)
