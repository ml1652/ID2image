import numpy as np
import os
from glob import glob

import numpy as np
import torch
import matplotlib.pyplot as plt

from models.image_to_latent import VGGToHist
from utilities.images import load_images
from tqdm import tqdm
from models.latent_optimizer import VGGFaceProcessing
from models.vgg_face2 import resnet50_scratch_dag
# image_path = r"C:\Users\Mingrui\Desktop\datasets\StyleGANimge_corp\webimage_alignmentTest\alignment_img14.jpg"
#
# reference_image = load_images(
#         [image_path]
#     )
#
# reference_image = torch.from_numpy(reference_image).cuda().detach()
#
# hist = torch.histc(reference_image)
# hist = hist.detach().cpu().numpy()
# plt.hist(hist ,bins='auto', density=True)
# plt.show()

loss_type = 'MSE'
#loss_type = 'EMD'
num_trainsets = 47800
Inserver = bool(os.environ['IN_SERVER']) if ('IN_SERVER' in os.environ) else False
epochs = int(os.environ['EPOCHES']) if ('EPOCHES' in os.environ) else 5000
validation_loss = 0.0

bins_num = 10
N_HIDDENS = int(os.environ['N_HIDDENS']) if ('N_HIDDENS' in os.environ) else 2
N_NEURONS =  int(os.environ['N_NEURONS']) if ('N_NEURONS' in os.environ) else 8
#lr=0.00001

lr=0.000001
if Inserver == True:
    import matplotlib
    matplotlib.use('Agg')
    directory = "./Croped_StyleGAN_Datasets/align_size(224,224)_move(0.250,0.000)_face_factor(0.800)_jpg/data/"
else:
    directory = "./Croped_StyleGAN_Datasets/align_size(224,224)_move(0.250,0.000)_face_factor(0.800)_jpg/data/"
    #directory = "C:/Users/Mingrui/Desktop/Github/pytorch_stylegan_encoder/InterFaceGAN/dataset_directory/"
filenames = sorted(glob(directory + "*.jpg"))
dlatents = np.load(directory + "wp.npy")
final_dlatents = []
for i in filenames :
    name = int(os.path.splitext(os.path.basename(i))[0])
    # name = int((i.split('\\')[-1]).split('.')[0])
    final_dlatents.append(dlatents[name])
dlatents = np.array(final_dlatents)
train_dlatents = dlatents[0:num_trainsets]
validation_dlatents = dlatents[num_trainsets:]

# ##################################################
def generate_vgg_descriptors(filenames):
    if Inserver == True:
        vgg_face_dag = resnet50_scratch_dag(
            '/scratch/staff/ml1652/StyleGAN_Reconstuction_server/Pretrained_model/resnet50_scratch_dag.pth').cuda().eval()
    else:
        vgg_face_dag = resnet50_scratch_dag('./resnet50_scratch_dag.pth').cuda().eval()
    descriptors = []
    vgg_processing = VGGFaceProcessing()
    for i in filenames:
        image = load_images([i]) #image value: (0,256)
        image = torch.from_numpy(image).cuda() #image value: (0,256)
        image = vgg_processing(image)  # vgg16: the vlaue between -2.04 - 2.54,dim = [1,3,256,256]
        feature = vgg_face_dag(image).cpu().detach().numpy()
        descriptors.append(feature) #descriptor[128, 28, 28] pool5_7x7_s1:[2048,1,1]
    return np.concatenate(descriptors, axis=0)

#np.save(save_path, np.concatenate(descriptors, axis=0))
#######################################################

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
        #self.flatten = torch.nn.Flatten()

    def forward(self, x):
        #b, n,= x.size()
        #self.centers = self.centers.unsqueeze(0).repeat(n,m,1).unsqueeze(3)
        #x = x - self.centers
        #x = self.flatten(x)
        x = torch.unsqueeze(x, 0) - torch.unsqueeze(self.centers, 1)
        x = torch.sigmoid(self.sigma * (x + self.delta / 2)) - torch.sigmoid(self.sigma * (x - self.delta / 2))
        x = x.sum(dim=1)
        #
        # x = torch.unsqueeze(x, 0) - torch.unsqueeze(self.centers, 1)
        # x = torch.exp(-0.5 * (x / self.sigma) ** 2) / (self.sigma * np.sqrt(np.pi * 2)) * self.delta
        # x = x.sum(dim=1)
        return x

# ##################################################
def plot_hist(hist_soft, histc,filenames):
    hist_soft = list(hist_soft)
    histc = list(histc)
    #plt.plot(hist_soft, color="red", linestyle='*',label = 'hist_soft' )
    #plt.hist(hist_soft, bins=bins, facecolor="red", edgecolor="red", alpha=0.5,)

    #plt.plot(histc, color="green", linestyle='-', label ='histc')
    #plt.hist(histc, bins=bins, facecolor="green", edgecolor="green", alpha=0.5,)

    subdiagram = 2

    plt.subplot(subdiagram, 1,1)
    plt.title('torch.histc')
    plt.xlabel('bin_num')
    plt.ylabel('pixel_num')
    plt.plot(histc, color="green", linestyle='-', label ='histc')
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

        image = load_images([i]) #image value: (0,256)
        image = torch.from_numpy(image).cuda() #image value: (0,256)
        r = image[:,0,:]
        g = image[:,1,:]
        b = image[:,2,:]

        hist_r = torch.histc(r.float(),bins, min=0, max=255).cpu().detach().numpy()
        hist_g = torch.histc(g.float(),bins, min=0, max=255).cpu().detach().numpy()
        hist_b = torch.histc(b.float(),bins, min=0, max=255).cpu().detach().numpy()

        # r = r.flatten()
        # g = g.flatten()
        # b = b.flatten()
        # softhist = SoftHistogram(bins, min=0, max=255, sigma=3*25).cuda()
        # hist_r = softhist(r).cpu().detach().numpy()
        # hist_g = softhist(g).cpu().detach().numpy()
        # hist_b = softhist(b).cpu().detach().numpy()


        hist = []
        num_pix = 224*224
        hist.append(hist_r/num_pix)
        hist.append(hist_g/num_pix)
        hist.append(hist_b/num_pix)

        hist_list.append(hist)
        #plot the comparison between histc and SoftHistogram
        #plot_hist(hist_r/num_pix,hist_r_histc/num_pix, i)


    hist_list= np.asarray(hist_list)
    #return np.concatenate(hist_list, axis=0)
    return hist_list
#np.save(save_path, np.concatenate(descriptors, axis=0))

def EMDLoss(output,target):
    # output and target must have size nbatch * nhist * nbins
    # We will compute an EMD for each histogram for each batch element
    loss = torch.zeros(output.shape[0], output.shape[1], output.shape[2]+1).cuda()  #loss: [batch_size, 3, bins_num]
    for i in range(1,output.shape[2]+1):
        loss[:,:,i] = output[:,:,i-1] + loss[:,:,i-1] - target[:,:,i-1] #loss:[32,3,20]
    # Compute the EMD
    loss = loss.abs().sum(dim=2) # loss: [32,3]
    # Sum over histograms
    loss = loss.sum(dim=1) #loss: [32]
    # Average over batch
    loss = loss.mean()
    return loss

# ######################################################

hist_file = directory + 'images_hist_bins=' + str(bins_num)+'.npy'
descriptor_file = directory + 'descriptors.npy'

if os.path.isfile(hist_file):
    hists = np.load(directory +'images_hist_bins=' + str(bins_num)+'.npy')

else:
    hists = generate_image_hist(filenames,bins_num)
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

train_dataset = hist_Dataset(train_descriptors, train_hist)
validation_dataset = hist_Dataset(validation_descriptors, validation_hist)

train_generator = torch.utils.data.DataLoader(train_dataset, batch_size=32)
validation_generator = torch.utils.data.DataLoader(validation_dataset, batch_size=32)

vgg_to_hist = VGGToHist(bins_num,BN = True, N_HIDDENS = N_HIDDENS,N_NEURONS = N_NEURONS).cuda()
#print(vgg_to_hist)

optimizer = torch.optim.Adam(vgg_to_hist.parameters(),lr)
if loss_type =='MSE':
    criterion = torch.nn.MSELoss()

class earth_mover_distance_(torch.nn.Module):
    def _init_(self):
        super().__init__()


    def forward(self,y_pred,y_true):
        y_t = torch.cumsum(y_true, -1)
        y_p = torch.cumsum(y_pred, -1)
        loss = torch.square(y_t - y_p)
        loss = torch.mean(loss, -1)
        return loss


progress_bar = tqdm(range(epochs))
Loss_list = []
Traning_loss =[]
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
            elif loss_type =='EMD':
                loss = EMDLoss(pred_hist, hists)
            validation_loss += loss.item()

    validation_loss = validation_loss / i
    Validation_loss.append(validation_loss)
    progress_bar.set_description(
        "Step: {0}, Loss: {1:4f}, Validation Loss: {2:4f}".format(i, running_loss / i, validation_loss))

# x, y
y1 = Traning_loss
y2 = Validation_loss
plt.subplot(2, 1, 1)
plt.plot( y1, 'o-')
plt.title('training loss vs. epoches')
plt.ylabel('training loss')
plt.subplot(2, 1, 2)
plt.plot( y2, '.-')
plt.xlabel('validation loss vs. epoches')
plt.ylabel('validation loss')
save_dir = "./diagram/vgg_to_Histogram_accuracy_loss_lr=" +str(lr)+ str(bins_num) + '_N_HIDDENS='+str(N_HIDDENS) + '_NEURONS='+str(N_NEURONS)+".jpg"
plt.savefig(save_dir)
#plt.show()

if  Inserver == True:
    torch.save(vgg_to_hist.state_dict(), '/scratch/staff/ml1652/StyleGAN_Reconstuction_server/vgg_to_hist_bins=' +str(bins_num) + '_N_HIDDENS='+str(N_HIDDENS) + '_NEURONS='+str(N_NEURONS)+'.pt')
else:
    torch.save(vgg_to_hist.state_dict(), './Trained_model/vgg_to_hist_bins=' + str(bins_num) +'_HIDDENS='+str(N_HIDDENS) + '_NEURONS='+str(N_NEURONS)+'.pt')



