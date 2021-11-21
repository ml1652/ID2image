import torch
import math
import numpy as np
import pandas as pd
import torch.nn.functional as F
import statistics
from glob import glob
from utilities.images import load_images
from models.latent_optimizer import VGGFaceProcessing
from models.vgg_face2 import resnet50_scratch_dag
from models.regressors import ImageToLandmarks_batch, VGGToHist, LandMarksRegressor, CelebaRegressor

vgg_face_dag = resnet50_scratch_dag(
    './Trained_model/resnet50_scratch_dag.pth').cuda().eval()
image_directory = './celeba/data/'
filenames = sorted(glob(image_directory + "*.jpg"))


def feed_into_Vgg(generated_image):
    features = vgg_face_dag(generated_image)
    return features


def image_to_Vggencoder(img):
    vgg_processing = VGGFaceProcessing()
    generated_image = vgg_processing(img)
    features = feed_into_Vgg(generated_image)
    return features


def feed_into_Image_to_landmarksRegressor(img):
    landmark_regressor = ImageToLandmarks_batch(landmark_num=68).cuda().eval()
    weights_path = './Trained_model/Image_to_landmarks_Regressor_batchnorm_lr=0.001.pt'
    landmark_regressor.load_state_dict(torch.load(weights_path))

    target_size = 64
    img = F.interpolate(img, target_size, mode='bilinear')
    pred_landmarks = landmark_regressor(img)
    return pred_landmarks


def generate_image_hist(image, bins=20):
    hist_list = []

    r = image[:, 0, :]
    g = image[:, 1, :]
    b = image[:, 2, :]

    hist_r = torch.histc(r.float(), bins, min=0, max=255).cpu().detach().numpy()
    hist_g = torch.histc(g.float(), bins, min=0, max=255).cpu().detach().numpy()
    hist_b = torch.histc(b.float(), bins, min=0, max=255).cpu().detach().numpy()

    hist = []
    num_pix = 224 * 224
    hist.append(hist_r / num_pix)
    hist.append(hist_g / num_pix)
    hist.append(hist_b / num_pix)
    hist_list.append(hist)
    hist_list = np.asarray(hist_list)
    return hist_list


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

    def forward(self, x):
        x = torch.unsqueeze(x, 0) - torch.unsqueeze(self.centers, 1)
        x = torch.sigmoid(self.sigma * (x + self.delta / 2)) - torch.sigmoid(self.sigma * (x - self.delta / 2))
        x = x.sum(dim=1)
        return x


def image_to_hist(croped_image, bins_num):
    r = croped_image[:, 0, :]
    g = croped_image[:, 1, :]
    b = croped_image[:, 2, :]

    softhist = SoftHistogram(bins_num, min=0, max=255, sigma=1.85).cuda()  # sigma=0.04
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


# calcualte the Earth mover's distance
def EMDLoss(output, target):
    # output and target must have size nbatch * nhist * nbins
    # We will compute an EMD for each histogram for each batch element
    loss = torch.zeros(output.shape[0], output.shape[1], output.shape[2] + 1)  # loss: [batch_size, 3, bins_num]
    for i in range(1, output.shape[2] + 1):
        loss[:, :, i] = output[:, :, i - 1] + loss[:, :, i - 1] - target[:, :, i - 1]  # loss:[32,3,20]
    # Compute the EMD
    loss = loss.abs().sum(dim=2)  # loss: [32,3]
    # Sum over histograms
    loss = loss.sum(dim=1)  # loss: [32]
    # Average over batch
    loss = loss.mean()
    return loss


# test the correct rate of lable predicgtion by the regressor on the CelebA test set.
def Celeba_Regressor_Result_preddistribution():
    label = 'Wearing_Hat'  # Eyeglasses,Smiling,Mouth_Slightly_Open, Blurry, Wearing_Hat, Wearing_Necktie
    vgg_processing = VGGFaceProcessing()
    celeba_regressor = CelebaRegressor().cuda()
    celeba_regressor.load_state_dict(torch.load('./Trained_model/celebaregressor_' + label + '.pt'))
    celeba_regressor.eval()

    pred_list = []
    for i in filenames[-10000:]:
        image = load_images([i])
        image = torch.from_numpy(image).cuda()
        image = vgg_processing(image)
        vgg_descriptors = vgg_face_dag(image).cuda()
        pred = celeba_regressor(vgg_descriptors)
        pred_choice = (pred > 0.4).int()
        pred_choice = torch.squeeze(pred_choice)
        pred_list.append(pred_choice)

    attribute_count = 0
    for number in pred_list:
        if number == 1:
            attribute_count += 1
    guess = attribute_count / 10000  #
    save_path = "./Result/Celeba_Regressor_Realdistribution" + label + ".txt"
    with open(save_path, 'w') as f:
        f.write(str(guess))

    data = pd.read_csv('./celeba/Anno/list_attr_celeba_name+glass+smiling_hat.csv').to_dict()
    droped_table = pd.read_csv('./celeba/Anno/imglist_after_crop.csv')
    leaved_img_name = droped_table['File_Name']

    count = 0
    for i in leaved_img_name[-10000:]:
        i = i.split('.')[0]
        i = int(i) - 1
        result = data[label][i]
        if result == 1:
            count += 1

    attribute_testsets = count / 10000
    save_path = "./Result/Celeba_Regressor_Result_preddistribution" + label + ".txt"
    with open(save_path, 'w') as f:
        f.write(str(attribute_testsets))


# calculate the mean normalized error between the target 68-landmark points and the predicted 68-landmark points
def computeMNELandmark(target_landmark, pred_landmark):
    sum = 0
    count = 0
    for target, pred in zip(target_landmark, pred_landmark):
        for i in range(0, len(target), 2):
            pred_x = int(pred[i])
            pred_y = int(pred[i + 1])
            target_x = int(target[i])
            target_y = int(target[i + 1])

            point_pred = np.array([pred_x, pred_y])
            point_target = np.array([target_x, target_y])
            L2 = np.linalg.norm(point_target - point_pred)
            sum += L2

            count += 1
            # choice the corner of the eye to compute the inter_ocular_distance
            if count == 37:
                left_cornereye_x = pred_x
                left_cornereye_y = pred_y
            elif count == 46:
                right_cornereye_x = pred_x
                right_cornereye_y = pred_y

    x = left_cornereye_x - right_cornereye_x
    y = left_cornereye_y - right_cornereye_y
    inter_ocular_distance = math.sqrt((x ** 2) + (y ** 2))
    # Mean normalised error
    MNE = sum / (68 * inter_ocular_distance)

    return MNE


# Calcualte the MNE between the facial landmarks predicted by featutres_to_landmark_regressor and dlib facial landmark detector
def landmarkRegressor_test(calculate_guess):
    landmarks_num = 68
    landmark_file_path = image_directory + "/tformed_landmark_68point.txt"
    orignal_landmark_data = pd.read_csv(
        landmark_file_path,
        sep=' ',
        header=None,
    )
    orignal_landmark_data = orignal_landmark_data.to_numpy()
    keys = orignal_landmark_data[:, 0].tolist()
    values = np.genfromtxt(landmark_file_path, dtype=np.float, usecols=range(1, 68 * 2 + 1)).tolist()
    target_landmarks = dict(zip(keys, values))

    from_ID_MNE_list = []
    from_Img_MNE_list = []
    for i in filenames[-10000:]:

        # from_ID
        reference_image = load_images([i])  # numpy
        reference_image = torch.from_numpy(reference_image).cuda()

        if calculate_guess == False:
            img_name = i.split("\\")[-1]
            target_landmark = torch.from_numpy(np.array(target_landmarks[img_name])).unsqueeze(0)
            vgg_features = image_to_Vggencoder(reference_image)
            features_to_landmarkRegressor = LandMarksRegressor(landmarks_num).cuda().eval()
            features_to_landmarkRegressor.load_state_dict(torch.load(
                './Trained_model/Celeba_Regressor_68_landmarks.pt'))

            pred_landmarks = features_to_landmarkRegressor(vgg_features)
            from_ID_MNE = computeMNELandmark(target_landmark, pred_landmarks)  # torch.Size([1, 136])
            from_ID_MNE_list.append(from_ID_MNE)
        else:
            target_landmark = np.load(
                "./average_landmark.npy")
            target_landmark = torch.from_numpy(target_landmark).unsqueeze(0)
        # from-image
        landmark_regressor = ImageToLandmarks_batch(landmark_num=68).cuda().eval()
        weights_path = './Trained_model/Image_to_landmarks_Regressor_batchnorm_lr=0.001.pt'
        landmark_regressor.load_state_dict(torch.load(weights_path))
        pred_landmarks = feed_into_Image_to_landmarksRegressor(
            reference_image.type(torch.float))
        pred_landmarks = pred_landmarks * (224 / 64)
        from_Img_MNE = computeMNELandmark(target_landmark, pred_landmarks)
        from_Img_MNE_list.append(from_Img_MNE)

    if calculate_guess == False:
        mean_from_ID_MNE = statistics.mean(from_ID_MNE_list)
        mean_from_Img_MNE = statistics.mean(from_Img_MNE_list)

        save_path = "./Result/LandmarkMNE.txt"
        with open(save_path, 'w') as f:
            f.write('mean_from_ID_MNE:' + str(mean_from_ID_MNE))
            f.write('mean_from_Img_MNE:' + str(mean_from_Img_MNE))
    else:
        mean_from_Img_MNE = statistics.mean(from_Img_MNE_list)
        save_path = "./Result/LandmarkGuess.txt"
        with open(save_path, 'w') as f:
            f.write('mean_from_LandmarkGuess:' + str(mean_from_Img_MNE))


# Test the performance of the image_to_histogram regressor
def Histregressor_Test():
    calculate_guess = True
    Loss_Type = 'EMD'  # or MSE

    from_ID_Hist_list = []
    from_Img_Hist_list = []

    bins_num = 10
    N_HIDDENS = 2
    N_NEURONS = 8
    features_to_hist = VGGToHist(bins_num, BN=True, N_HIDDENS=N_HIDDENS, N_NEURONS=N_NEURONS).cuda().eval()
    hist_file_name = 'vgg_to_hist_bins=' + str(bins_num) + '_HIDDENS=' + str(N_HIDDENS) + '_NEURONS=' + str(
        N_NEURONS) + '.pt'
    features_to_hist.load_state_dict(torch.load(
        './Trained_model/' + hist_file_name))
    criterion = torch.nn.MSELoss()
    for i in filenames[-10000:]:
        reference_image = load_images([i])  # numpy
        reference_image = torch.from_numpy(reference_image).cuda()
        vgg_features = image_to_Vggencoder(reference_image)

        if calculate_guess == True:
            Target_Hist = np.load("C:/Users/Mingrui/Desktop/Result_for_Paper/3.1Result/average_histogram.npy")
            Target_Hist = torch.from_numpy(Target_Hist).unsqueeze(0).detach().cpu()
        else:
            Target_Hist = generate_image_hist(reference_image, bins_num)
            Target_Hist = torch.from_numpy(Target_Hist).detach().cpu()
            From_ID_to_Hist = features_to_hist(vgg_features).detach().cpu()
            if Loss_Type == 'EMD':
                from_ID_EMD = EMDLoss(From_ID_to_Hist, Target_Hist)  # From_ID_to_Hist([1, 3, 10])
            elif Loss_Type == 'MSE':
                from_ID_EMD = criterion(From_ID_to_Hist, Target_Hist)
            from_ID_EMD = from_ID_EMD.numpy()  # array(6.2224045, dtype=float32)
            from_ID_Hist_list.append(from_ID_EMD)

        # SoftHist
        From_Img_to_HIst = image_to_hist(reference_image, bins_num)
        From_Img_to_HIst = From_Img_to_HIst.unsqueeze(0).detach().cpu()

        if Loss_Type == 'EMD':
            from_Img_EMD = EMDLoss(From_Img_to_HIst, Target_Hist)  # torch.Size([1, 136])
        elif Loss_Type == 'MSE':
            from_Img_EMD = criterion(From_Img_to_HIst, Target_Hist)

        from_Img_EMD = from_Img_EMD.numpy()
        from_Img_Hist_list.append(from_Img_EMD)
    if calculate_guess == True:
        mean_from_Img_EMD = sum(from_Img_Hist_list) / len(from_Img_Hist_list)

        save_path = "./Result/" + "Result_table_HistGuess" + Loss_Type + ".txt"
        with open(save_path, 'w') as f:
            f.write('HistGuess:' + str(mean_from_Img_EMD))
    else:
        mean_from_Img_EMD = sum(from_Img_Hist_list) / len(from_Img_Hist_list)
        mean_from_ID_EMD = sum(from_ID_Hist_list) / len(from_ID_Hist_list)

        save_path = "./Result/" + "Result_table_Hist" + Loss_Type + ".txt"
        with open(save_path, 'w') as f:
            f.write('mean_from_ID:' + str(mean_from_ID_EMD))
            f.write('mean_from_Img:' + str(mean_from_Img_EMD))


# Calculate the average value of the landamrk points and image's histogram in datasets
def calcuateMeanvalue():
    label_type = "Histogram"  # "Histogram"，"Landmark"
    landmark_file_path = image_directory + "/tformed_landmark_68point.txt"

    if label_type == "Landmark":
        orignal_landmark_data = pd.read_csv(
            landmark_file_path,
            sep=' ',
            header=None,
        )
        orignal_landmark_data = orignal_landmark_data.to_numpy()
        keys = orignal_landmark_data[:, 0].tolist()
        values = np.genfromtxt(landmark_file_path, dtype=np.float, usecols=range(1, 68 * 2 + 1)).tolist()
        target_landmarks = dict(zip(keys, values))

        training_landmarks_list = []
        for i in filenames[:-10000]:
            img_name = i.split("\\")[-1]
            training_target_landmarks = np.array(target_landmarks[img_name])
            training_landmarks_list.append(training_target_landmarks)

        training_landmarks_list = np.array(training_landmarks_list)
        average = np.mean(training_landmarks_list, axis=0)
        save_path = "./Result/" + "average_landmark" + ".npy"
        np.save(save_path, average)
    elif label_type == "Histogram":
        bins_num = 10
        Trainig_Hist_list = []
        for i in filenames[:-10000]:
            reference_image = load_images([i])
            reference_image = torch.from_numpy(reference_image).cuda()
            Traning_Hist = generate_image_hist(reference_image, bins_num)
            Traning_Hist = np.squeeze(Traning_Hist)
            Trainig_Hist_list.append(Traning_Hist)

        average = np.mean(Trainig_Hist_list, axis=0)
        save_path = "./Result/" + "average_histogram" + ".npy"
        np.save(save_path, average)


def Celeba_Regressor_Result_Guess():
    label = 'Wearing_Hat'  # Eyeglasses,Smiling,Mouth_Slightly_Open, Blurry, Wearing_Hat, Wearing_Necktie

    vgg_processing = VGGFaceProcessing()
    celeba_regressor = CelebaRegressor().cuda()
    celeba_regressor.load_state_dict(
        torch.load('./Trained_model/celebaregressor_' + label + '.pt'))
    celeba_regressor.eval()
    droped_table = pd.read_csv('./celeba/Anno/imglist_after_crop.csv')
    data = pd.read_csv('./celeba/Anno/list_attr_celeba_name+glass+smiling_hat.csv').to_dict()
    leaved_img_name = droped_table['File_Name']

    pred_list = []
    correct_count = 0
    for i in leaved_img_name[-10000:]:
        image = load_images([image_directory + i])
        image = torch.from_numpy(image).cuda()
        image = vgg_processing(image)
        vgg_descriptors = vgg_face_dag(image).cuda()
        pred = celeba_regressor(vgg_descriptors)
        # pred = str(pred)
        pred_choice = (pred > 0.4).int()
        pred_choice = torch.squeeze(pred_choice)
        pred_list.append(pred_choice)

        # real label
        i = i.split('.')[0]
        i = int(i) - 1
        real = data[label][i]
        if pred_choice == real:
            correct_count += 1
    guess = correct_count / 10000
    save_path = "./Result/" + "3.4Result_table" + label + "_Guess" + ".txt"
    with open(save_path, 'w') as f:
        f.write(str(guess))
