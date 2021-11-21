import torch.nn.functional as F
import torch


class PostSynthesisProcessing(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.min_value = -1
        self.max_value = 1

    def forward(self, synthesized_image):
        synthesized_image = (synthesized_image - self.min_value) * torch.tensor(255).float() / (
                self.max_value - self.min_value)
        synthesized_image = torch.clamp(synthesized_image + 0.5, min=0, max=255)

        return synthesized_image


class VGGProcessing(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.image_size = 256
        self.mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(-1, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(-1, 1, 1)

    def forward(self, image):
        image = image / torch.tensor(255).float()
        image = image.float()
        image = F.adaptive_avg_pool2d(image, self.image_size)
        image = (image - self.mean) / self.std

        return image


class VGGFaceProcessing(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.image_size = 224
        self.std = torch.tensor([1, 1, 1], device="cuda").view(-1, 1, 1)
        self.mean = torch.tensor([131.45376586914062, 103.98748016357422, 91.46234893798828], device="cuda").view(-1, 1,
                                                                                                                  1)

    def forward(self, image):
        image = image.float()
        if image.shape[2] != 224 or image.shape[3] != 224:
            image = F.adaptive_avg_pool2d(image, self.image_size)
        image = (image - self.mean) / self.std
        return image
