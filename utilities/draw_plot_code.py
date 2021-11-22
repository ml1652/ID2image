import cv2
import numpy as np
import matplotlib as plt
import torch

def CompareLandmarkRegreeesorOnTargetImage(img, target_landmark, pred_landmark, save_file_name, learning_rate,
                                           weight_landmarkloss, iterations, image_path):
    point_size = 1
    target_point_color = (0, 255, 0)
    pred_point_color = (0, 0, 255)
    thickness = 4
    image_resize = 224

    str1 = './' + save_file_name + '/TargetImg_LandmarkPlot_lr=' + str(
        learning_rate) + '_b=' + str(weight_landmarkloss) + '_iter=' + str(iterations) + '_'
    str2 = (image_path).split('/')[-1]
    landmarks_image_save_path = str1 + str2

    img = img.detach().cpu().numpy().squeeze().transpose(1, 2, 0)

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


def draw_targetLandmarkPoint(img, target_landmark, save_file_name, learning_rate, weight_landmarkloss, iterations,
                             image_path):
    point_size = 1
    target_point_color = (0, 255, 0)
    thickness = 4

    str1 = './' + save_file_name + '/TargetLandmarkPlot__lr=' + str(
        learning_rate) + '_b=' + str(weight_landmarkloss) + '_iter=' + str(iterations) + '_'
    str2 = (image_path).split('/')[-1]
    landmarks_image_save_path = str1 + str2

    img = img.detach().cpu().numpy().squeeze().transpose(1, 2, 0)
    image_draw = img
    image_draw = np.uint8(image_draw)
    image_resize = 224
    image_draw = cv2.resize(image_draw, (image_resize, image_resize), interpolation=cv2.INTER_AREA)
    image_draw = cv2.cvtColor(image_draw, cv2.COLOR_BGR2RGB)

    for landmark in target_landmark:
        for i in range(0, len(landmark), 2):
            point1 = int(landmark[i])
            point2 = int(landmark[i + 1])
            point_tuple = (point1, point2)
            image_draw = cv2.circle(image_draw, point_tuple, point_size, target_point_color, thickness)
    cv2.imwrite(landmarks_image_save_path, image_draw)


def draw_RegressedLandmarkPoint(img, pred_landmark, save_file_name, learning_rate, weight_landmarkloss, iterations,
                                image_path):
    point_size = 1
    target_point_color = (0, 0, 255)
    thickness = 4

    str1 = './' + save_file_name + '/RegressedLandmarkPlot__lr=' + str(
        learning_rate) + '_b=' + str(weight_landmarkloss) + '_iter=' + str(iterations) + '_'
    str2 = (image_path).split('/')[-1]
    landmarks_image_save_path = str1 + str2

    img = img.detach().cpu().numpy().squeeze().transpose(1, 2, 0)

    image_draw = img
    image_draw = np.uint8(image_draw)
    image_resize = 224
    image_draw = cv2.resize(image_draw, (image_resize, image_resize), interpolation=cv2.INTER_AREA)
    image_draw = cv2.cvtColor(image_draw, cv2.COLOR_BGR2RGB)

    for landmark in pred_landmark:
        for i in range(0, len(landmark), 2):
            point1 = int(landmark[i])
            point2 = int(landmark[i + 1])
            point_tuple = (point1, point2)
            image_draw = cv2.circle(image_draw, point_tuple, point_size, target_point_color, thickness)
    cv2.imwrite(landmarks_image_save_path, image_draw)


def plotloss(Landmark_loss_plot, Id_loss_plot, hist_l_plot, save_file_name, learning_rate, weight_landmarkloss,
             iterations,
             image_path):
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

    str1 = './' + save_file_name + '/Threeloss_plot__lr=' + str(
        learning_rate) + '_b=' + str(weight_landmarkloss) + '_iter=' + str(iterations) + '_'
    str2 = (image_path).split('/')[-1]

    hist_save_path = str1 + str2
    plt.savefig(hist_save_path)
    plt.close('all')


# Comparison of the image's histogram between the target image and the reconstructed image
def draw_histogram_Compare(reference_img, generated_img, bins_num, save_file_name, learning_rate,
                           weight_landmarkloss,
                           iterations,
                           image_path):
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

    subdiagram = 3
    plt.subplot(subdiagram, 1, 1)
    plt.plot(hist_target[0], color="red", linestyle='-')
    plt.plot(hist_pred[0], color="red", linestyle=':')
    plt.xticks([])
    plt.yticks([])

    plt.subplot(subdiagram, 1, 2)
    plt.plot(hist_target[1], color="green", linestyle='-')
    plt.plot(hist_pred[1], color="green", linestyle=':')
    plt.xticks([])
    plt.yticks([])

    values = np.arange(0, 10)
    plt.subplot(subdiagram, 1, 3)
    plt.plot(hist_target[2], color="blue", linestyle='-')
    plt.plot(hist_pred[2], color="blue", linestyle=':')
    plt.xticks(values)
    ax = plt.gca()
    labels = [item.get_text() for item in ax.get_xticklabels()]
    labels[0] = '0'
    labels[-1] = '255'
    ax.set_xticklabels(labels)
    plt.yticks([])

    str1 = './' + save_file_name + '/Histogram_histc_TargetAndGeneratedImgCompare__lr=' + str(
        learning_rate) + '_b=' + str(weight_landmarkloss) + '_iter=' + str(iterations) + '_'
    str2 = (image_path).split('/')[-1]
    str2 = str2.split('.')[0]

    hist_save_path = str1 + str2 + '.pdf'
    plt.savefig(hist_save_path)
    plt.close('all')
    return
