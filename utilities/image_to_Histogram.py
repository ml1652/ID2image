import torch


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

    softhist = SoftHistogram(bins_num, min=0, max=255, sigma=1.85).cuda()
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
