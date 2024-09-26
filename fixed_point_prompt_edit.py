from __future__ import annotations
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Maps GPU 2 in your system to cuda:0 in your program

import argparse
from PIL import Image
import torch
import numpy as np
from src.enums import Model_Type, Scheduler_Type
from src.utils.enums_utils import model_type_to_size, get_pipes
from src.config import RunConfig
from main import run as run_model, create_noise_list
import pandas as pd
import datasets
from tqdm import tqdm
from src.metrics.metric_util import calculate_metrics, calculate_clip_score
from torchvision import transforms
from io import BytesIO
from pytorch_fid import fid_score

import ast
from torchvision.transforms.functional import to_pil_image
# Command-line argument parsing
parser = argparse.ArgumentParser(description='Run Model Experiment')
parser.add_argument('--fixed_point_iterations', type=int, default=2, help='Number of fixed point iterations (default: 2)')
parser.add_argument('--fixed_point_inversion_steps', type=int, default=2, help='Number of fixed point inversion steps (default: 2)')
parser.add_argument('--num_renoise_steps', type=int, default=9, help='Number of renoise steps (default: 9)')
parser.add_argument('--dataset_config_name', type=str, default='2_add_object_80', help="Dataset configuration name (default: '1_change_object_80')")

# MIN_PROMPT_LENGTH = 0
# MAX_PROMPT_LENGTH = 35

args = parser.parse_args()

# Setup
seed = 7865
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_type = Model_Type.SDXL_Turbo
scheduler_type = Scheduler_Type.EULER
image_size = model_type_to_size(Model_Type.SDXL_Turbo)
pipe_inversion, pipe_inference = get_pipes(model_type, scheduler_type, device=device)
generator = torch.Generator().manual_seed(seed)
noise_list = create_noise_list(model_type, 50, generator=generator)
edit_cfg = 1.0

# Configuration based on arguments
config = RunConfig(seed=seed, fixed_point_iterations=args.fixed_point_iterations,
                   fixed_point_inversion_steps=args.fixed_point_inversion_steps,
                   num_renoise_steps=args.num_renoise_steps)

torch.manual_seed(config.seed)
np.random.seed(seed=config.seed)

# Dataset loading
ds = datasets.load_dataset("UB-CVML-Group/PIE_Bench_pp", args.dataset_config_name, split='V1')

to_tensor_transform = transforms.Compose([transforms.ToTensor()])

# DataFrame to store metrics
metrics_list = []
config_str = (f"seed_{config.seed}_fpi_{config.fixed_point_iterations}_"
              f"fpis_{config.fixed_point_inversion_steps}_renoise_steps_{config.num_renoise_steps}_"
              f"dataset_{args.dataset_config_name}")



# Directory setup for FID calculation
real_image_dir = f'./fid_images/real/{config_str}/'
edited_image_dir = f'./fid_images/edited/{config_str}/'
os.makedirs(real_image_dir, exist_ok=True)
os.makedirs(edited_image_dir, exist_ok=True)
# Iterate through the dataset
for idx, sample in enumerate(tqdm(ds, total=len(ds))):
    original_image = sample['image']
    if isinstance(original_image, dict):
        original_image = Image.open(BytesIO(original_image['bytes']))
    
    # Resize the image if necessary
    if original_image.size[0] > 420 and original_image.size[1] > 420:
        original_image = original_image.resize((512, 512))
    else:
        continue

    source_prompt = sample['source_prompt']
    prompts = [[source_prompt]]
    display_prompts = [source_prompt]

    # if source_prompt < MIN_PROMPT_LENGTH or source_prompt > MAX_PROMPT_LENGTH:
    #     continue
    # Iterate over each edit action in the sample's edit_action dictionary
    for edit_key, edit_info in ast.literal_eval(sample['edit_action']).items():
        position = edit_info['position']
        edit_type = edit_info['edit_type']
        action = edit_info['action']
        words = source_prompt.split()
        edit_key_words = edit_key.split()

        # Depending on the action, modify the prompt
        if action == '+':  # Add the word(s) at the specified position
            words.insert(position, edit_key)
            display_prompts.append(f'+{edit_key}')
        elif action == '-':  # Remove the word(s) at the specified position
            if words[position:position + len(edit_key_words)] == edit_key_words:
                del words[position:position + len(edit_key_words)]
                display_prompts.append(f'-{edit_key}')
            else:
                continue  # Position mismatch
        else:  # Replace the word(s) at the specified position with the action
            action_words = action.split()
            if words[position:position + len(action_words)] == action_words:
                del words[position:position + len(action_words)]
                words[position:position] = edit_key_words
                display_prompts.append(f'{action}->{edit_key}')
            else:
                continue  # Position mismatch

        new_prompt = ' '.join(words)
        prompts.append([new_prompt])

        # Run the model
        x_0_hat_edit, inv_latent, noise_list, _, _ = run_model(
            original_image, source_prompt, config, latents=None, edit_prompt=new_prompt,
            pipe_inversion=pipe_inversion, pipe_inference=pipe_inference,
            noise=noise_list, edit_cfg=edit_cfg, do_reconstruction=True
        )

        x_0_hat, _, _, _, _ = run_model(
            original_image, source_prompt, config, latents=inv_latent,
            pipe_inversion=pipe_inversion, pipe_inference=pipe_inference,
            noise=noise_list, edit_cfg=edit_cfg, do_reconstruction=True
        )

        # Convert the result to tensor
        x_0_hat_edit_im = to_tensor_transform(x_0_hat_edit).to(device)
        x_0_hat_im = to_tensor_transform(x_0_hat).to(device)
        original_image.save(os.path.join(real_image_dir, f"real_{idx}.png"))
        to_pil_image(x_0_hat_edit_im).save(os.path.join(edited_image_dir, f"edit_{idx}.png"))
        # Calculate metrics
        metrics = calculate_metrics(original_image, x_0_hat_im)
        metrics['clip_score'] = calculate_clip_score(x_0_hat_edit_im.unsqueeze(0), [new_prompt])
        metrics['source_prompt'] = source_prompt
        metrics['edit_prompt'] = new_prompt
        # Append metrics to list
        metrics_list.append(metrics)

        # Save metrics every 100 samples
        if (idx + 1) % 100 == 0:
            metrics_df = pd.DataFrame(metrics_list)
            metrics_df.to_csv(f"./metrics_reports/edit2/metrics_checkpoint_{config_str}.csv", index=False)

# After dataset processing, calculate FID for all images
fid_value = fid_score.calculate_fid_given_paths([real_image_dir, edited_image_dir], batch_size=50, device=device, dims=2048)
print(f"FID Score: {fid_value}")

# Convert the list of metrics to a DataFrame
metrics_df = pd.DataFrame(metrics_list)
metrics_df['fid'] = fid_value
# Save the final metrics
metrics_df.to_csv(f"./metrics_reports/edit2/final_metrics_{config_str}.csv", index=False)

# Calculate and print the mean of each metric
# mean_metrics = metrics_df.mean()
# print("Mean of each metric:")
# print(mean_metrics)
