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

# Initialize CLIP model and processor
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(dev)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def calculate_l2_difference(image1, image2, device='cuda'):
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

def calculate_clip(image, text, device='cuda'):
    # Preprocess the image and text
    inputs = clip_processor(text=[text], images=image, return_tensors="pt", padding=True).to(device)
    
    # Get logits from the CLIP model
    outputs = clip_model(**inputs)
    
    # Get the logits per image
    logits_per_image = outputs.logits_per_image
    
    # Divide by 100 to get the value as a percentage-like score (optional, depends on your intent)
    probs = logits_per_image / 100.0
    
    # Extract the scalar value from the tensor using .item()
    return probs.item()

def calculate_metrics(image1, image2, edit_prompt=None, device='cuda', size=(512, 512)):
    if isinstance(image1, Image.Image):
        image1 = image1.resize(size)
        image1 = to_tensor_transform(image1).to(device)
    if isinstance(image2, Image.Image):
        image2 = image2.resize(size)
        image2 = to_tensor_transform(image2).to(device)
        
    l2 = calculate_l2_difference(image1, image2, device)
    psnr = calculate_psnr(image1, image2, device)
    lpips = calculate_lpips(image1, image2, device)
    
    metrics = {"l2": l2, "psnr": psnr, "lpips": lpips}
    
    # Calculate CLIP score only if edit_prompt is provided
    if edit_prompt:
        clip_score = calculate_clip(image2, edit_prompt, device)
        metrics["clip_score"] = clip_score
    
    return metrics

def get_empty_metrics():
    return {"l2": 0, "psnr": 0, "lpips": 0, "clip_score": 0}

def print_results(results):
    clip_score = results.get('clip_score', 'N/A')
    print(f"Reconstruction Metrics: L2: {results['l2']},\t PSNR: {results['psnr']},\t LPIPS: {results['lpips']},\t CLIP Score: {clip_score}")
