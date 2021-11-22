import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from glob import glob
from models.regressors import VGGToHist
from utilities.images import load_images
from tqdm import tqdm
from models.latent_optimizer import VGGFaceProcessing
from models.vgg_face2 import resnet50_scratch_dag


def generate_vgg_descriptors(filenames):
    vgg_face_dag = resnet50_scratch_dag('./Trained_model/resnet50_scratch_dag.pth').cuda().eval()
    descriptors = []
    vgg_processing = VGGFaceProcessing()
    for i in filenames:
        image = load_images([i])
        image = torch.from_numpy(image).cuda()
        image = vgg_processing(image)
        feature = vgg_face_dag(image).cpu().detach().numpy()
        descriptors.append(feature)
    return np.concatenate(descriptors, axis=0)


def plot_hist(hist_soft, histc, filenames):
    hist_soft = list(hist_soft)
    histc = list(histc)
    subdiagram = 2

    plt.subplot(subdiagram, 1, 1)
    plt.title('torch.histc')
    plt.xlabel('bin_num')
    plt.ylabel('pixel_num')
    plt.plot(histc, color="green", linestyle='-', label='histc')
    plt.subplot(subdiagram, 1, 2)
    plt.subplots_adjust(wspace=0, hspace=0.5)
    plt.title('hist_soft')
    plt.xlabel('bin_num')
    plt.ylabel('pixel_num')
    plt.plot(hist_soft, color="red", linestyle='-', label='hist_soft')

    path_ = './diagram/plot_histc_'
    img_name = filenames.split('\\')[-1]
    plot_path = path_ + img_name
    plt.savefig(plot_path)
    plt.close('all')


def generate_image_hist(filenames, bins=20):
    hist_list = []
    for i in filenames:
        image = load_images([i])  # image value: (0,256)
        image = torch.from_numpy(image).cuda()  # image value: (0,256)
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


def EMDLoss(output, target):
    loss = torch.zeros(output.shape[0], output.shape[1], output.shape[2] + 1).cuda()  # loss: [batch_size, 3, bins_num]
    for i in range(1, output.shape[2] + 1):
        loss[:, :, i] = output[:, :, i - 1] + loss[:, :, i - 1] - target[:, :, i - 1]  # loss:[32,3,20]
    loss = loss.abs().sum(dim=2)  # loss: [32,3]
    # Sum over histograms
    loss = loss.sum(dim=1)  # loss: [32]
    # Average over batch
    loss = loss.mean()
    return loss


class hist_Dataset(torch.utils.data.Dataset):
    def __init__(self, hists, dlatents):
        self.hists = hists
        self.dlatents = dlatents

    def __len__(self):
        return len(self.hists)

    def __getitem__(self, index):
        hists = self.hists[index]
        dlatent = self.dlatents[index]

        return hists, dlatent


loss_type = 'MSE'
num_trainsets = 47800
Inserver = bool(os.environ['IN_SERVER']) if ('IN_SERVER' in os.environ) else False
epochs = int(os.environ['EPOCHES']) if ('EPOCHES' in os.environ) else 5000
validation_loss = 0.0

bins_num = 10
N_HIDDENS = int(os.environ['N_HIDDENS']) if ('N_HIDDENS' in os.environ) else 2
N_NEURONS = int(os.environ['N_NEURONS']) if ('N_NEURONS' in os.environ) else 8

lr = 0.000001

directory = "./datasets/"
filenames = sorted(glob(directory + "*.jpg"))
dlatents = np.load(directory + "wp.npy")
final_dlatents = []
for i in filenames:
    name = int(os.path.splitext(os.path.basename(i))[0])
    final_dlatents.append(dlatents[name])
dlatents = np.array(final_dlatents)
train_dlatents = dlatents[0:num_trainsets]
validation_dlatents = dlatents[num_trainsets:]
hist_file = directory + 'images_hist_bins=' + str(bins_num) + '.npy'
descriptor_file = directory + 'descriptors.npy'

if os.path.isfile(hist_file):
    hists = np.load(directory + 'images_hist_bins=' + str(bins_num) + '.npy')

else:
    hists = generate_image_hist(filenames, bins_num)
    np.save(hist_file, hists)

if os.path.isfile(descriptor_file):
    descriptor = np.load(directory + "descriptors.npy")
else:
    descriptor = generate_vgg_descriptors(filenames)
    np.save(descriptor_file, descriptor)

train_descriptors = descriptor[0:num_trainsets]
validation_descriptors = descriptor[num_trainsets:]
train_hist = hists[0:num_trainsets]
validation_hist = hists[num_trainsets:]
train_dataset = hist_Dataset(train_descriptors, train_hist)
validation_dataset = hist_Dataset(validation_descriptors, validation_hist)
train_generator = torch.utils.data.DataLoader(train_dataset, batch_size=32)
validation_generator = torch.utils.data.DataLoader(validation_dataset, batch_size=32)
vgg_to_hist = VGGToHist(bins_num, BN=True, N_HIDDENS=N_HIDDENS, N_NEURONS=N_NEURONS).cuda()
optimizer = torch.optim.Adam(vgg_to_hist.parameters(), lr)
if loss_type == 'MSE':
    criterion = torch.nn.MSELoss()
progress_bar = tqdm(range(epochs))
Loss_list = []
Traning_loss = []
Validation_loss = []
for epoch in progress_bar:
    running_loss = 0.0
    vgg_to_hist.train()

    for i, (vgg_descriptors, hists) in enumerate(train_generator, 1):
        optimizer.zero_grad()
        vgg_descriptors, hists = vgg_descriptors.cuda(), hists.cuda()
        pred_hist = vgg_to_hist(vgg_descriptors)
        if loss_type == 'MSE':
            loss = criterion(pred_hist, hists)
        elif loss_type == 'EMD':
            loss = EMDLoss(pred_hist, hists)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        progress_bar.set_description(
            "Step: {0}, Loss: {1:4f}, Validation Loss: {2:4f}".format(i, running_loss / i, validation_loss))

    traning_loss = running_loss / i
    Traning_loss.append(traning_loss)
    validation_loss = 0.0

    vgg_to_hist.eval()
    for i, (vgg_descriptors, hists) in enumerate(validation_generator, 1):
        with torch.no_grad():
            vgg_descriptors, hists = vgg_descriptors.cuda(), hists.cuda()
            pred_hist = vgg_to_hist(vgg_descriptors)
            if loss_type == 'MSE':
                loss = criterion(pred_hist, hists)
            elif loss_type == 'EMD':
                loss = EMDLoss(pred_hist, hists)
            validation_loss += loss.item()

    validation_loss = validation_loss / i
    Validation_loss.append(validation_loss)
    progress_bar.set_description(
        "Step: {0}, Loss: {1:4f}, Validation Loss: {2:4f}".format(i, running_loss / i, validation_loss))

y1 = Traning_loss
y2 = Validation_loss
plt.subplot(2, 1, 1)
plt.plot(y1, 'o-')
plt.title('training loss vs. epoches')
plt.ylabel('training loss')
plt.subplot(2, 1, 2)
plt.plot(y2, '.-')
plt.xlabel('validation loss vs. epoches')
plt.ylabel('validation loss')
save_dir = "./diagram/vgg_to_Histogram_accuracy_loss_lr=" + str(lr) + str(bins_num) + '_N_HIDDENS=' + str(
    N_HIDDENS) + '_NEURONS=' + str(N_NEURONS) + ".jpg"
plt.savefig(save_dir)

torch.save(vgg_to_hist.state_dict(),
           './Trained_model/vgg_to_hist_bins=' + str(bins_num) + '_HIDDENS=' + str(N_HIDDENS) + '_NEURONS=' + str(
               N_NEURONS) + '.pt')
