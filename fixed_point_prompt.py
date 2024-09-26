from __future__ import annotations

from PIL import Image
import torch
import numpy as np
from src.enums import Model_Type, Scheduler_Type
from src.utils.enums_utils import model_type_to_size, get_pipes
from src.config import RunConfig
from main import run as run_model, create_noise_list
import torch
import pandas as pd
import os
from datasets import load_dataset
from tqdm import tqdm

def diffuse_latent(z_T, src_prompt):
    z_0, _, _, _, __ = run_model(None,
                                    src_prompt,
                                    config,
                                    latents=z_T,
                                    pipe_inversion=pipe_inversion,
                                    pipe_inference=pipe_inference,
                                    noise=noise_list,
                                    edit_cfg=edit_cfg,
                                    do_reconstruction=True,
                                    return_image_latent=True)
    z_0 = z_0.detach().cpu()
    x_0, _, _, _, __ = run_model(None,
                                    src_prompt,
                                    config,
                                    latents=z_T,
                                    pipe_inversion=pipe_inversion,
                                    pipe_inference=pipe_inference,
                                    noise=noise_list,
                                    edit_cfg=edit_cfg,
                                    do_reconstruction=True,
                                    return_image_latent=False)
    return z_0, x_0

seed = 7865
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_type = Model_Type.SDXL_Turbo
scheduler_type = Scheduler_Type.EULER
image_size = model_type_to_size(Model_Type.SDXL_Turbo)
pipe_inversion, pipe_inference = get_pipes(model_type, scheduler_type, device=device)

generator = torch.Generator().manual_seed(seed)


noise_list = create_noise_list(model_type, 4, generator=generator)

# tgt_prompt = "a lego kitten is sitting in a basket on a branch" #tgt_prompt
inv_latent = None
# noise_list = None
edit_cfg = 1.0


config = RunConfig(seed=seed, fixed_point_iterations=0, fixed_point_inversion_steps = 0, num_renoise_steps=9)
config_prompt_aware = RunConfig(seed = seed, fixed_point_iterations=2, fixed_point_inversion_steps = 2, num_renoise_steps=9)
torch.manual_seed(config.seed)
np.random.seed(seed=config.seed)
z_T = torch.randn(1, 4, 64, 64).to(device)




# Number of samples to process
samples = 10000
save_every = 100
output_file = "l2_metrics_with_prompts_no_renoise.csv"
checkpoint_file = "checkpoint_with_prompts_no_renoise.csv"

# Load the dataset
dataset = load_dataset("andyyang/stable_diffusion_prompts_2m", split=f"train[:{samples}]")

# Load the previous checkpoint if exists
if os.path.exists(checkpoint_file):
    checkpoint_df = pd.read_csv(checkpoint_file)
    start_idx = int(checkpoint_df['last_processed_idx'].iloc[-1]) + 1
    metrics_df = pd.read_csv(output_file)
else:
    start_idx = 0
    metrics_df = pd.DataFrame(columns=['idx', 'prompt', 'L2_z_0_E', 'L2_z_0_tilde', 'L2_consistency'])

# Resume or start processing
for idx in tqdm(range(start_idx, len(dataset))):
    sample = dataset[idx]
    
    z_T = torch.randn(1, 4, 64, 64).to(device)
    prompt = sample['text']

    # Diffuse latent and run models
    z_0, x_0 = diffuse_latent(z_T, prompt)
    
    z_0_hat_E, _, _, _, _ = run_model(x_0,
                                      prompt,
                                      config,
                                      latents=None,
                                      pipe_inversion=pipe_inversion,
                                      pipe_inference=pipe_inference,
                                      noise=noise_list,
                                      edit_cfg=edit_cfg,
                                      do_reconstruction=True,
                                      return_image_latent=True)
    
    z_0_tilde_after_inv, _, _, all_latents, all_fixed_point_latents = run_model(x_0,
                                    prompt,
                                    config_prompt_aware,
                                    latents=None,
                                    pipe_inversion=pipe_inversion,
                                    pipe_inference=pipe_inference,
                                    noise=noise_list,
                                    edit_cfg=edit_cfg,
                                    do_reconstruction=True,
                                    return_image_latent=True)
    
    z_0_E = all_fixed_point_latents[0][0]
    # Calculate L2 metrics
    L2_z_0_E = torch.norm(z_0-z_0_hat_E.cpu(), p=2).item()
    L2_z_0_tilde = torch.norm(z_0-z_0_tilde_after_inv.cpu(), p=2).item()
    z_0_tilde = all_latents[0]
    L2_consistency = torch.norm(z_0_hat_E-z_0_E, p=2).item()
    L2_consistency_fp = torch.norm(z_0_tilde-z_0_tilde_after_inv, p=2).item()

    # Create a temporary DataFrame for the current sample
    temp_df = pd.DataFrame([{
        'idx': idx,
        'prompt': prompt,
        'L2_z_0_E': L2_z_0_E,
        'L2_z_0_tilde': L2_z_0_tilde,
        'L2_consistency': L2_consistency,
        'L2_consistency_fp': L2_consistency_fp
    }])

    # Use pd.concat to append the temporary DataFrame to the main metrics DataFrame
    metrics_df = pd.concat([metrics_df, temp_df], ignore_index=True)
    
    # Save metrics every 10 samples
    if (idx + 1) % save_every == 0 or (idx + 1) == len(dataset):
        # Save the DataFrame to CSV file
        metrics_df.to_csv(output_file, index=False)
        
        # Save the current checkpoint as a CSV file
        checkpoint_df = pd.DataFrame({'last_processed_idx': [idx]})
        checkpoint_df.to_csv(checkpoint_file, index=False)
        
        print(f"Checkpoint saved at sample {idx + 1}")
        