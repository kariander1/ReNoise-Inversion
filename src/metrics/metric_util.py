import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
from src.metrics.lpips import LPIPS
from torchmetrics.functional.multimodal import clip_score
from functools import partial

dev = 'cuda'
to_tensor_transform = transforms.Compose([transforms.ToTensor()])
mse_loss = nn.MSELoss()
clip_score_fn = partial(clip_score, model_name_or_path="openai/clip-vit-base-patch16")


def calculate_clip_score(images, prompts):
    # If the images are in range [0, 1], scale them to [0, 255] and convert to uint8
    images_uint8 = (images * 255).to(torch.uint8)
    
    # Ensure images are in the correct shape Bx3x512x512, which they already are
    clip_score = clip_score_fn(images_uint8, prompts).detach()
    
    # Return the rounded clip score
    return round(float(clip_score), 4)

def calculate_l2_difference(image1, image2, device = 'cuda'):
    if isinstance(image1, Image.Image):
        image1 = to_tensor_transform(image1).to(device)
    if isinstance(image2, Image.Image):
        image2 = to_tensor_transform(image2).to(device)

    mse = mse_loss(image1, image2).item()
    return mse

def calculate_psnr(image1, image2, device = 'cuda'):
    max_value = 1.0
    if isinstance(image1, Image.Image):
        image1 = to_tensor_transform(image1).to(device)
    if isinstance(image2, Image.Image):
        image2 = to_tensor_transform(image2).to(device)
    
    mse = mse_loss(image1, image2)
    psnr = 10 * torch.log10(max_value**2 / mse).item()
    return psnr


loss_fn = LPIPS(net_type='vgg').to(dev).eval()

def calculate_lpips(image1, image2, device = 'cuda'):
    if isinstance(image1, Image.Image):
        image1 = to_tensor_transform(image1).to(device)
    if isinstance(image2, Image.Image):
        image2 = to_tensor_transform(image2).to(device)
    
    loss = loss_fn(image1, image2).item()
    return loss

def calculate_metrics(image1, image2, device = 'cuda', size=(512, 512)) -> dict:
    if isinstance(image1, Image.Image):
        image1 = image1.resize(size)
        image1 = to_tensor_transform(image1).to(device)
    if isinstance(image2, Image.Image):
        image2 = image2.resize(size)
        image2 = to_tensor_transform(image2).to(device)
        
    l2 = calculate_l2_difference(image1, image2, device)
    psnr = calculate_psnr(image1, image2, device)
    lpips = calculate_lpips(image1, image2, device)
    return {"l2": l2, "psnr": psnr, "lpips": lpips}

def get_empty_metrics():
    return {"l2": 0, "psnr": 0, "lpips": 0}

def print_results(results):
    print(f"Reconstruction Metrics: L2: {results['l2']},\t PSNR: {results['psnr']},\t LPIPS: {results['lpips']}")