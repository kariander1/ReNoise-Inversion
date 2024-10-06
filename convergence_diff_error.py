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
import torch
import pandas as pd
import datasets
from tqdm import tqdm
from src.metrics.metric_util import calculate_metrics
from torchvision import transforms

# Setup
seed = 7865
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_type = Model_Type.SDXL_Turbo
scheduler_type = Scheduler_Type.EULER
image_size = model_type_to_size(Model_Type.SDXL_Turbo)
pipe_inversion, pipe_inference = get_pipes(model_type, scheduler_type, device=device)
generator = torch.Generator().manual_seed(seed)
noise_list = create_noise_list(model_type, 4, generator=generator)
edit_cfg = 1.0
num_renoise_steps = 20
average_step_range = (0, 20)
# Configuration
config = RunConfig(seed=seed, 
                   noise_regularization_lambda_ac=0, noise_regularization_lambda_kl=0,num_renoise_steps=num_renoise_steps,average_step_range=average_step_range)

torch.manual_seed(config.seed)
np.random.seed(seed=config.seed)

# Dataset loading
ds = datasets.load_dataset("ydshieh/coco_dataset_script", "2017", data_dir="/home/dcor/shaiyehezkel/ReNoise-Inversion1/", split='validation')

to_tensor_transform = transforms.Compose([transforms.ToTensor()])

# List of epsilon values to iterate over
epsilon_values = [0.3,0.5,1,2,3,5]  # Add more values as needed

# Iterate over epsilon values
for epsilon in epsilon_values:
    # DataFrame to store metrics
    metrics_list = []
    config_str = f"iter_error_diff_epsilon{epsilon}"

    # Iterate through the dataset
    for idx, sample in enumerate(tqdm(ds, total=len(ds))):
        image_path = sample['image_path']
        prompt = sample['caption']
        original_image = Image.open(image_path).convert("RGB")

        # Resize the image if necessary
        if original_image.size[0] > 420 and original_image.size[1] > 420:
            original_image = original_image.resize((512, 512))
        else:
            continue

        # Testing:
        edit_prompt = prompt

        # Run the model
        img, inv_latent, noise, all_latents, _, mean_inversions = run_model(original_image,
                                    prompt,
                                    config,
                                    latents=None,
                                    edit_prompt=edit_prompt,
                                    pipe_inversion=pipe_inversion,
                                    pipe_inference=pipe_inference,
                                    noise=noise_list,
                                    edit_cfg=edit_cfg,
                                    do_reconstruction=True,
                                    inversion_step_type = "error_diff",
                                    epsilon=epsilon)

        # Calculate metrics
        x_0_hat_im = to_tensor_transform(img).to(device)

        metrics = calculate_metrics(original_image, x_0_hat_im)
        
            # Add the mean_inversions value to the metrics
        metrics['mean_inversions'] = mean_inversions

        # Append metrics to list
        metrics_list.append(metrics)

        # Save metrics every 100 samples
        # if (idx + 1) % 100 == 0:
        # # if (idx + 1) % 1 == 0:
        #     metrics_df = pd.DataFrame(metrics_list)
        #     metrics_df.to_csv(f"./metrics_reports/{config_str}.csv", index=False)

    # Convert the list of metrics to a DataFrame
    # Create DataFrame from the metrics list
    metrics_df = pd.DataFrame(metrics_list)

    # Calculate the mean of each metric
    mean_metrics = metrics_df.mean()

    # Convert the mean metrics to a DataFrame and set an appropriate label (e.g., "mean")
    mean_df = pd.DataFrame(mean_metrics).T  # Transpose to have the same format as metrics_df
    mean_df['description'] = 'mean'  # Add a column to label the mean row

    # Add the mean row to the original DataFrame
    metrics_df['description'] = 'data'  # Add a column to label the data rows
    metrics_df = pd.concat([metrics_df, mean_df], ignore_index=True)

    # Save the updated DataFrame to a CSV file
    metrics_df.to_csv(f"./metrics_reports/metrics_{config_str}.csv", index=False)

    # Print the mean metrics
    print(f"Mean of each metric for epsilon {epsilon}:")
    print(mean_metrics)

