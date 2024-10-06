from __future__ import annotations
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Maps GPU 2 in your system to cuda:0 in your program

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
from src.metrics.metric_util import calculate_metrics
from torchvision import transforms
from io import BytesIO
import ast
from torchvision.transforms.functional import to_pil_image

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
num_renoise_steps = 9
average_step_range = (0, 20)

# Dataset options and epsilon values
dataset_options = [
    "1_change_object_80", "2_add_object_80", "3_delete_object_80", "4_change_attribute_content_40",
    "5_change_attribute_pose_40", "6_change_attribute_color_40", "7_change_attribute_material_40",
    "8_change_background_80", "9_change_style_80"
]
epsilon_values = [1,10,20,30,40,45,50,55,65]  # Add more values as needed

# Configuration
config = RunConfig(seed=seed, num_renoise_steps=num_renoise_steps, average_step_range=average_step_range)
torch.manual_seed(config.seed)
np.random.seed(seed=config.seed)

# Iterate over epsilon values
for epsilon in epsilon_values:
    print(f"Processing epsilon: {epsilon}")
    
    # DataFrame to store combined metrics for all datasets
    combined_metrics_list = []

    # Iterate over dataset options
    for dataset_option in dataset_options:
        print(f"Processing dataset option: {dataset_option}")
        
        # Dataset loading (change split and dataset as per your need)
        ds = datasets.load_dataset("UB-CVML-Group/PIE_Bench_pp", dataset_option, split='V1')

        to_tensor_transform = transforms.Compose([transforms.ToTensor()])

        # Limit the number of samples to 20
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
            
            print('ast.literal_eval(sample[edit_action]).items(): ', ast.literal_eval(sample['edit_action']).items())

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
                x_0_hat_edit, inv_latent, noise_list, _, _, mean_inversions = run_model(
                    original_image, source_prompt, config, latents=None, edit_prompt=new_prompt,
                    pipe_inversion=pipe_inversion, pipe_inference=pipe_inference,
                    noise=noise_list, edit_cfg=edit_cfg, do_reconstruction=True,inversion_step_type = "error",
                    epsilon=epsilon
                )

                # Calculate metrics
                x_0_hat_im = to_tensor_transform(x_0_hat_edit).to(device)

                metrics = calculate_metrics(original_image, x_0_hat_im)
                # metrics = calculate_metrics(original_image, x_0_hat_edit, new_prompt)
                
                metrics['mean_inversions'] = mean_inversions

                # Append metrics to combined list for all datasets
                combined_metrics_list.append(metrics)

                # Convert the list of metrics to a DataFrame and calculate the running mean
                metrics_df = pd.DataFrame(combined_metrics_list)
                running_mean_metrics = metrics_df.mean()

                # Print running mean metrics
                print(f"Running mean of each metric after {idx + 1} samples for dataset {dataset_option} and epsilon {epsilon}:")
                print(running_mean_metrics)

    # Convert the combined list of metrics to a DataFrame
    combined_metrics_df = pd.DataFrame(combined_metrics_list)

    # Save the combined metrics for this epsilon
    combined_metrics_df.to_csv(f"./metrics_reports/edit_error_epsilon_{epsilon}.csv", index=False)

    # Calculate the mean of all metrics and print it
    final_mean_metrics = combined_metrics_df.mean()

    # Convert the mean metrics to a DataFrame and set an appropriate label (e.g., "mean")
    mean_df = pd.DataFrame(final_mean_metrics).T  # Transpose to have the same format as the metrics DataFrame
    mean_df['description'] = 'mean'  # Add a column to label the mean row

    # Add the mean row to the original DataFrame
    combined_metrics_df['description'] = 'data'  # Add a column to label the data rows
    combined_metrics_df = pd.concat([combined_metrics_df, mean_df], ignore_index=True)

    # Save the updated DataFrame to a CSV file
    combined_metrics_df.to_csv(f"./metrics_reports/final_edit_error_epsilon_{epsilon}.csv", index=False)

    # Print the final mean metrics
    print(f"Final mean of each metric for epsilon {epsilon}:")
    print(final_mean_metrics)
