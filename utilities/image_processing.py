import torch
import torch.nn.functional as F


def crop_image_forVGGFACE2(croped_image):
    target_size = 224
    croped_image = croped_image[:, :, 210:906, 164:860]
    croped_image = F.interpolate(croped_image, target_size, mode='bilinear')
    return croped_image


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


def post_porcessing_outputImg(img):
    post_synthesis_processing = PostSynthesisProcessing()
    generated_image = post_synthesis_processing(img)
    return generated_image


class VGGFaceProcessing(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.image_size = 224
        self.std = torch.tensor([1, 1, 1], device="cuda").view(-1, 1, 1)
        self.mean = torch.tensor([131.0912, 103.8827, 91.4953], device="cuda").view(-1, 1, 1)

    def forward(self, image):
        image = image.float()
        if image.shape[2] != 224 or image.shape[3] != 224:
            image = F.adaptive_avg_pool2d(image, self.image_size)
        image = (image - self.mean) / self.std
        return image
