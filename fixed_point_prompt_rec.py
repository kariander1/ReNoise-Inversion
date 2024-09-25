from __future__ import annotations
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Maps GPU 2 in your system to cuda:0 in your program

from PIL import Image
import torch
import numpy as np
from src.enums import Model_Type, Scheduler_Type
from src.utils.enums_utils import model_type_to_size, get_pipes
from src.config import RunConfig
from main import run as run_model, create_noise_list
import torch
import pandas as pd
import datasets
from tqdm import tqdm
from src.metrics.metric_util import calculate_metrics
from torchvision import transforms
from main import run as invert

# Setup
seed = 7865
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_type = Model_Type.SDXL_Turbo
# scheduler_type = Scheduler_Type.EULER if model_type==Model_Type.SDXL_Turbo else Scheduler_Type.DDIM
scheduler_type = Scheduler_Type.EULER
image_size = model_type_to_size(model_type)
pipe_inversion, pipe_inference = get_pipes(model_type, scheduler_type, device=device)
# pipe_inversion_fp, pipe_inference_fp = get_pipes(model_type, scheduler_type, device=device)
generator = torch.Generator().manual_seed(seed)
noise_list = create_noise_list(model_type, 4, generator=generator)
edit_cfg = 1.0
                    
# Configuration
config = RunConfig(
    seed=seed,
    fixed_point_iterations=2,
    fixed_point_inversion_steps=2,
    num_inference_steps = 4 if model_type==Model_Type.SDXL_Turbo else 50,
    num_inversion_steps = 4 if model_type==Model_Type.SDXL_Turbo else 50,
    num_renoise_steps = 9 if model_type==Model_Type.SDXL_Turbo else 1,
    perform_noise_correction = True if model_type==Model_Type.SDXL_Turbo else False,
    noise_regularization_num_reg_steps = 0,
    model_type=model_type,
    scheduler_type = scheduler_type)

# config_fp = RunConfig(
#     seed=seed,
#     fixed_point_iterations=0,
#     fixed_point_inversion_steps=0,
#     num_inference_steps = 4 if model_type==Model_Type.SDXL_Turbo else 50,
#     num_inversion_steps = 4 if model_type==Model_Type.SDXL_Turbo else 50,
#     num_renoise_steps = 9 if model_type==Model_Type.SDXL_Turbo else 1,
#     perform_noise_correction = True if model_type==Model_Type.SDXL_Turbo else False,
#     noise_regularization_num_reg_steps = 0,
#     model_type=model_type,
#     scheduler_type = scheduler_type)


torch.manual_seed(config.seed)
np.random.seed(seed=config.seed)

# Dataset loading
ds = datasets.load_dataset("ydshieh/coco_dataset_script", "2017", data_dir="/home/dcor/shaiyehezkel/ReNoise-Inversion1/", split='validation')

to_tensor_transform = transforms.Compose([transforms.ToTensor()])

# DataFrame to store metrics
metrics_list = []
config_str = f"seed_{config.seed}_fpi_{config.fixed_point_iterations}_fpis_{config.fixed_point_inversion_steps}"

# Iterate through the dataset
for idx, sample in enumerate(tqdm(ds, total=len(ds))):
    image_path = sample['image_path']
    prompt = sample['caption']
    original_image = Image.open(image_path).convert("RGB")
    
    # Resize the image if necessary
    if original_image.size[0] > 420 and original_image.size[1] > 420:
        original_image = original_image.resize(image_size)
    else:
        continue
    
    _, inv_latent, _, all_latents, _ = invert(original_image,
                                        prompt,
                                        config,
                                        # latents=z_0_tilde,
                                        pipe_inversion=pipe_inversion,
                                        pipe_inference=pipe_inference,
                                        do_reconstruction=False)

    x_0_hat = pipe_inference(image = inv_latent,
                            prompt = prompt,
                            denoising_start=0.0,
                            num_inference_steps = config.num_inference_steps,
                            guidance_scale = 1.0).images[0]
    
    # Convert the result to tensor
    x_0_hat = to_tensor_transform(x_0_hat).to(device)
    
    # Calculate metrics
    metrics = calculate_metrics(original_image, x_0_hat)
    
    # Append metrics to list
    metrics_list.append(metrics)
    
    # Save metrics every 100 samples
    if (idx + 1) % 100 == 0:
        metrics_df = pd.DataFrame(metrics_list)
        metrics_df.to_csv(f"./metrics_reports/metrics_checkpoint_{config_str}.csv", index=False)

# Convert the list of metrics to a DataFrame
metrics_df = pd.DataFrame(metrics_list)

# Save the final metrics
metrics_df.to_csv(f"./metrics_reports/final_metrics_{config_str}.csv", index=False)

# Calculate and print the mean of each metric
mean_metrics = metrics_df.mean()
print("Mean of each metric:")
print(mean_metrics)
