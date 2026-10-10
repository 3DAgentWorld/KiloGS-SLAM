import random
import torch
from gaussian_splatting.utils.loss_utils import l1_loss, ssim

def compute_photometric_loss(viewpoint_cam, image):
    gt_image = viewpoint_cam.original_image.cuda()
    Ll1 = l1_loss(image, gt_image)
    loss = (1.0 - 0.2) * Ll1 + 0.2 * (1.0 - ssim(image.unsqueeze(0), gt_image.unsqueeze(0)))
    return loss

def sampling_cameras(my_viewpoint_stack, num_cams: int = 10):
    """Randomly sample up to ``num_cams`` camera objects from the stack/dict."""

    working_stack = list(my_viewpoint_stack.values())
    if len(working_stack) == 0:
        return []

    num_cams = min(num_cams, len(working_stack))
    return random.sample(working_stack, num_cams)

def get_loss(reconstructed_image, original_image):
    l1_loss = torch.mean(torch.abs(reconstructed_image - original_image), 0).detach()
    l1_loss_norm = (l1_loss - torch.min(l1_loss)) / (torch.max(l1_loss) - torch.min(l1_loss))

    return l1_loss_norm