from __future__ import annotations
import gradio as gr
import spaces
from PIL import Image
import torch

from src.enums import Model_Type, Scheduler_Type
from src.utils.enums_utils import model_type_to_size, get_pipes
from src.config import RunConfig
from main import run as run_model


DESCRIPTION = '''# ReNoise - Prompt Awareness & Convergence
This is a demo for our extension of the "ReNoise: Real Image Inversion Through Iterative Noising" [paper](https://garibida.github.io/ReNoise-Inversion/). Code is available [here](https://github.com/garibida/ReNoise-Inversion).
In this work, we expand on ReNoise by exploring convergence criteria for the iterative noising process and introducing prompt-aware adjustments to enhance the quality of edits. The ReNoise inversion technique is compatible with various diffusion models, including recent few-step models like SDXL-Turbo.
The demo performs real image editing using Prompt-Aware ReNoise inversion, where the input image is resized to 512x512, the optimal size for SDXL Turbo.
'''


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_type = Model_Type.SDXL_Turbo
scheduler_type = Scheduler_Type.EULER
image_size = model_type_to_size(Model_Type.SDXL_Turbo)
pipe_inversion, pipe_inference = get_pipes(model_type, scheduler_type, device=device)

cache_size = 30
prev_configs = [None for i in range(cache_size)]
prev_inv_latents = [None for i in range(cache_size)]
prev_images = [None for i in range(cache_size)]
prev_noises = [None for i in range(cache_size)]

@spaces.GPU
def main_pipeline(
        input_image: str,
        src_prompt: str,
        tgt_prompt: str,
        edit_cfg: float,
        seed: int,
        number_of_renoising_iterations: int,
        inersion_strength: float,
        average_latent_estimations: bool,
        first_step_range_start: int,
        first_step_range_end: int,
        rest_step_range_start: int,
        rest_step_range_end: int,
        noise_regularization_lambda_ac: float,
        noise_regularization_lambda_kl: float,
        perform_noise_correction: bool,
        fixed_point_iterations: int,
        fixed_point_inversion_steps: int ):

        global prev_configs, prev_inv_latents, prev_images, prev_noises

        first_step_range = (first_step_range_start, first_step_range_end)
        rest_step_range = (rest_step_range_start, rest_step_range_end)

        config = RunConfig(model_type = model_type,
                    num_inference_steps = 4,
                    num_inversion_steps = 4, 
                    guidance_scale = 0.0,
                    max_num_renoise_steps_first_step = first_step_range_end+1,
                    num_renoise_steps = number_of_renoising_iterations,
                    inversion_max_step = inersion_strength,
                    average_latent_estimations = average_latent_estimations,
                    average_first_step_range = first_step_range,
                    average_step_range = rest_step_range,
                    scheduler_type = scheduler_type,
                    noise_regularization_num_reg_steps = 4,
                    noise_regularization_num_ac_rolls = 5,
                    noise_regularization_lambda_ac = noise_regularization_lambda_ac,
                    noise_regularization_lambda_kl = noise_regularization_lambda_kl,
                    perform_noise_correction = perform_noise_correction,
                    seed = seed,
                    fixed_point_iterations = fixed_point_iterations,
                    fixed_point_inversion_steps = fixed_point_inversion_steps)
        
        inv_latent = None
        noise_list = None
        for i in range(cache_size):
            if prev_configs[i] is not None and prev_configs[i] == config and prev_images[i] == input_image:
                print(f"Using cache for config #{i}")
                inv_latent = prev_inv_latents[i]
                noise_list = prev_noises[i]
                prev_configs.pop(i)
                prev_inv_latents.pop(i)
                prev_images.pop(i)
                prev_noises.pop(i)
                break

        original_image = Image.open(input_image).convert("RGB").resize(image_size)

        res_image, inv_latent, noise, all_latents, all_fixed_point_latents = run_model(original_image,
                                    src_prompt,
                                    config,
                                    latents=inv_latent,
                                    pipe_inversion=pipe_inversion,
                                    pipe_inference=pipe_inference,
                                    edit_prompt=tgt_prompt,
                                    noise=noise_list,
                                    edit_cfg=edit_cfg,
                                    do_reconstruction=True)

        prev_configs.append(config)
        prev_inv_latents.append(inv_latent)
        prev_images.append(input_image)
        prev_noises.append(noise)
        
        if len(prev_configs) > cache_size:
            print("Popping cache")
            prev_configs.pop(0)
            prev_inv_latents.pop(0)
            prev_images.pop(0)
            prev_noises.pop(0)

        return res_image


with gr.Blocks(css='app/style.css') as demo:
    gr.Markdown(DESCRIPTION)

    gr.HTML(
        '''<a href="https://huggingface.co/spaces/garibida/ReNoise-Inversion?duplicate=true">
        <img src="https://bit.ly/3gLdBN6" alt="Duplicate Space"></a>Duplicate the Space to run privately without waiting in queue''')

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                label="Input image",
                type="filepath",
                height=image_size[0],
                width=image_size[1]
            )
            src_prompt = gr.Text(
                label='Source Prompt',
                max_lines=1,
                placeholder='A kitten is sitting in a basket on a branch',
            )
            tgt_prompt = gr.Text(
                label='Target Prompt',
                max_lines=1,
                placeholder='A plush toy kitten is sitting in a basket on a branch',
            )
            with gr.Accordion("Advanced Options", open=False):
                edit_cfg = gr.Slider(
                    label='Denoise Classifier-Free Guidence Scale',
                    minimum=1.0,
                    maximum=3.5,
                    value=1.0,
                    step=0.1
                )
                seed = gr.Slider(
                    label='Seed',
                    minimum=0,
                    maximum=16*1024,
                    value=7865,
                    step=1
                )
                number_of_renoising_iterations = gr.Slider(
                    label='Number of ReNoise Iterations',
                    minimum=0,
                    maximum=20,
                    value=9,
                    step=1
                )
                inersion_strength = gr.Slider(
                    label='Inversion Strength',
                    info="Indicates how much to invert the reference image. The number of denoising steps depends on the amount of noise initially added. When strength is 1, the image will be inverted to complete noise and the denoising process will run for the full number of steps (4). When strength is 0.5, the image will be inverted to half noise and the denoising process will run for 2 steps.",
                    minimum=0.0,
                    maximum=1.0,
                    value=1.0,
                    step=0.25
                )
                avg_gradients = gr.Checkbox(
                    label="Preform Estimation Averaging",
                    info="IMPROVES RECONSTRUCTION. Averagin the estination over multiple ReNoise iterations can improve the quality of the reconstruction. The Next 4 sliders control the range of steps to average over. The first two sliders control the range of steps to average over for the first inversion step (t < 250). The last two sliders control the range of steps to average over for the rest of the inversion step (t > 250).",
                    value=True
                )
                first_step_range_start = gr.Slider(
                    label='First Estimation in Average (t < 250)',
                    minimum=0,
                    maximum=21,
                    value=0,
                    step=1
                )
                first_step_range_end = gr.Slider(
                    label='Last Estimation in Average (t < 250)',
                    minimum=0,
                    maximum=21,
                    value=5,
                    step=1
                )
                rest_step_range_start = gr.Slider(
                    label='First Estimation in Average (t > 250)',
                    minimum=0,
                    maximum=21,
                    value=8,
                    step=1
                )
                rest_step_range_end = gr.Slider(
                    label='Last Estimation in Average (t > 250)',
                    minimum=0,
                    maximum=21,
                    value=10,
                    step=1
                )
                noise_regularization_num_reg_steps = 4
                noise_regularization_num_ac_rolls = 5
                noise_regularization_lambda_ac = gr.Slider(
                    label='Labmda AC',
                    info="IMPROVES EDITABILITY. The weight of the pair loss in the noise prediction regulariztion. This loss encourages the inversion to predict more editable noise. A higher value allows more significant changes to the image (higher editability), but may result in less faithful reconstructions.",
                    minimum=0.0,
                    maximum=50.0,
                    value=20.0,
                    step=1.0
                )
                noise_regularization_lambda_kl = gr.Slider(
                    label='Labmda Patch KL',
                    info="IMPROVES EDITABILITY. This weight controls the strength of the patch-level KL divergence term in the noise prediction regularization.  While it encourages editable noise like the Labmda AC, it often has a less detrimental effect on reconstruction fidelity.",
                    minimum=0.0,
                    maximum=0.4,
                    value=0.065,
                    step=0.005
                )
                noise_correction = gr.Checkbox(
                    label="Preform Noise Correction",
                    info="IMPROVES RECONSTRUCTION. Performs noise correction to improve the reconstruction of the image.",
                    value=True
                )
                fixed_point_iterations = gr.Slider(
                    label='Number of prompt-aware iterations',
                    info="IMPROVES EDITABILITY. This controls number of prompt-aware iterations to perform. The more iterations, the more the inversion will be aligned with the source prompt.",
                    minimum=0,
                    maximum=10,
                    value=2,
                    step=1
                )
                fixed_point_inversion_steps = gr.Slider(
                    label='Number of inversion steps per fixed point iteration',
                    minimum=1,
                    maximum=4,
                    value=2,
                    step=1
                )
            run_button = gr.Button('Edit')
        with gr.Column():
            # result = gr.Gallery(label='Result')
            result = gr.Image(
                label="Result",
                type="pil",
                height=image_size[0],
                width=image_size[1]
            )

            examples = [
                [
                    "example_images/space_dog.png", #input_image
                    "a dog wearing space suit", #src_prompt
                    "a cat wearing space suit", #tgt_prompt
                    1.0, #edit_cfg
                    7865, #seed
                    9, #number_of_renoising_iterations
                    1.0, #inersion_strength
                    True, #avg_gradients
                    0, #first_step_range_start
                    5, #first_step_range_end
                    8, #rest_step_range_start
                    10, #rest_step_range_end
                    20.0, #noise_regularization_lambda_ac
                    0.055, #noise_regularization_lambda_kl
                    True, #noise_correction,
                    0, #fixed_point_iterations
                    1, #fixed_point_inversion_steps
                ],
                [
                    "example_images/space_dog.png", #input_image
                    "a dog wearing space suit", #src_prompt
                    "a cat wearing space suit", #tgt_prompt
                    1.0, #edit_cfg
                    7865, #seed
                    9, #number_of_renoising_iterations
                    1.0, #inersion_strength
                    True, #avg_gradients
                    0, #first_step_range_start
                    5, #first_step_range_end
                    8, #rest_step_range_start
                    10, #rest_step_range_end
                    20.0, #noise_regularization_lambda_ac
                    0.055, #noise_regularization_lambda_kl
                    True, #noise_correction,
                    2, #fixed_point_iterations
                    2, #fixed_point_inversion_steps
                ],
                                [
                    "example_images/space_dog.png", #input_image
                    "a dog wearing space suit", #src_prompt
                    "a dog wearing space suit, mouth open", #tgt_prompt
                    1.0, #edit_cfg
                    7865, #seed
                    9, #number_of_renoising_iterations
                    1.0, #inersion_strength
                    True, #avg_gradients
                    0, #first_step_range_start
                    5, #first_step_range_end
                    8, #rest_step_range_start
                    10, #rest_step_range_end
                    20.0, #noise_regularization_lambda_ac
                    0.055, #noise_regularization_lambda_kl
                    True, #noise_correction,
                    0, #fixed_point_iterations
                    1, #fixed_point_inversion_steps
                ],
                [
                    "example_images/space_dog.png", #input_image
                    "a dog wearing space suit", #src_prompt
                    "a dog wearing space suit, mouth open", #tgt_prompt
                    1.0, #edit_cfg
                    7865, #seed
                    9, #number_of_renoising_iterations
                    1.0, #inersion_strength
                    True, #avg_gradients
                    0, #first_step_range_start
                    5, #first_step_range_end
                    8, #rest_step_range_start
                    10, #rest_step_range_end
                    20.0, #noise_regularization_lambda_ac
                    0.055, #noise_regularization_lambda_kl
                    True, #noise_correction,
                    2, #fixed_point_iterations
                    2, #fixed_point_inversion_steps
                ],
                [
                    "example_images/monkey.jpeg", #input_image
                    "a monkey sitting on a tree branch in the forest", #src_prompt
                    "a chicken sitting on a tree branch in the forest", #tgt_prompt
                    1.0, #edit_cfg
                    7865, #seed
                    9, #number_of_renoising_iterations
                    1.0, #inersion_strength
                    True, #avg_gradients
                    0, #first_step_range_start
                    5, #first_step_range_end
                    8, #rest_step_range_start
                    10, #rest_step_range_end
                    20.0, #noise_regularization_lambda_ac
                    0.055, #noise_regularization_lambda_kl
                    True, #noise_correction,
                    0, #fixed_point_iterations
                    1, #fixed_point_inversion_steps
                ],
                [
                    "example_images/monkey.jpeg", #input_image
                    "a monkey sitting on a tree branch in the forest", #src_prompt
                    "a chicken sitting on a tree branch in the forest", #tgt_prompt
                    1.0, #edit_cfg
                    7865, #seed
                    9, #number_of_renoising_iterations
                    1.0, #inersion_strength
                    True, #avg_gradients
                    0, #first_step_range_start
                    5, #first_step_range_end
                    8, #rest_step_range_start
                    10, #rest_step_range_end
                    20.0, #noise_regularization_lambda_ac
                    0.055, #noise_regularization_lambda_kl
                    True, #noise_correction,
                    2, #fixed_point_iterations
                    2, #fixed_point_inversion_steps
                ],
                [
                    "example_images/monkey.jpeg", #input_image
                    "a monkey sitting on a tree branch in the forest", #src_prompt
                    "a monkey sitting on a metal pole in the forest", #tgt_prompt
                    1.0, #edit_cfg
                    7865, #seed
                    9, #number_of_renoising_iterations
                    1.0, #inersion_strength
                    True, #avg_gradients
                    0, #first_step_range_start
                    5, #first_step_range_end
                    8, #rest_step_range_start
                    10, #rest_step_range_end
                    20.0, #noise_regularization_lambda_ac
                    0.055, #noise_regularization_lambda_kl
                    True, #noise_correction,
                    0, #fixed_point_iterations
                    1, #fixed_point_inversion_steps
                ],
                [
                    "example_images/monkey.jpeg", #input_image
                    "a monkey sitting on a tree branch in the forest", #src_prompt
                    "a monkey sitting on a metal pole in the forest", #tgt_prompt
                    1.0, #edit_cfg
                    7865, #seed
                    9, #number_of_renoising_iterations
                    1.0, #inersion_strength
                    True, #avg_gradients
                    0, #first_step_range_start
                    5, #first_step_range_end
                    8, #rest_step_range_start
                    10, #rest_step_range_end
                    20.0, #noise_regularization_lambda_ac
                    0.055, #noise_regularization_lambda_kl
                    True, #noise_correction,
                    2, #fixed_point_iterations
                    2, #fixed_point_inversion_steps
                ],
                [
                    "example_images/bicycle.jpg", #input_image
                    "a slanted mountain bicycle on the road in front of a building", #src_prompt
                    "a slanted mountain bicycle on the grass in front of a building", #tgt_prompt
                    1.0, #edit_cfg
                    7865, #seed
                    9, #number_of_renoising_iterations
                    1.0, #inersion_strength
                    True, #avg_gradients
                    0, #first_step_range_start
                    5, #first_step_range_end
                    8, #rest_step_range_start
                    10, #rest_step_range_end
                    20.0, #noise_regularization_lambda_ac
                    0.055, #noise_regularization_lambda_kl
                    True, #noise_correction,
                    0, #fixed_point_iterations
                    1, #fixed_point_inversion_steps
                ],
                [
                    "example_images/bicycle.jpg", #input_image
                    "a slanted mountain bicycle on the road in front of a building", #src_prompt
                    "a slanted mountain bicycle on the grass in front of a building", #tgt_prompt
                    1.0, #edit_cfg
                    7865, #seed
                    9, #number_of_renoising_iterations
                    1.0, #inersion_strength
                    True, #avg_gradients
                    0, #first_step_range_start
                    5, #first_step_range_end
                    8, #rest_step_range_start
                    10, #rest_step_range_end
                    20.0, #noise_regularization_lambda_ac
                    0.055, #noise_regularization_lambda_kl
                    True, #noise_correction,
                    3, #fixed_point_iterations
                    2, #fixed_point_inversion_steps
                ],
                                [
                    "example_images/bicycle.jpg", #input_image
                    "a slanted mountain bicycle on the road in front of a building", #src_prompt
                    "a slanted vespa on the grass in front of a building", #tgt_prompt
                    1.0, #edit_cfg
                    7865, #seed
                    9, #number_of_renoising_iterations
                    1.0, #inersion_strength
                    True, #avg_gradients
                    0, #first_step_range_start
                    5, #first_step_range_end
                    8, #rest_step_range_start
                    10, #rest_step_range_end
                    20.0, #noise_regularization_lambda_ac
                    0.055, #noise_regularization_lambda_kl
                    True, #noise_correction,
                    0, #fixed_point_iterations
                    1, #fixed_point_inversion_steps
                ],
                [
                    "example_images/bicycle.jpg", #input_image
                    "a slanted mountain bicycle on the road in front of a building", #src_prompt
                    "a slanted vespa on the grass in front of a building", #tgt_prompt
                    1.0, #edit_cfg
                    7865, #seed
                    9, #number_of_renoising_iterations
                    1.0, #inersion_strength
                    True, #avg_gradients
                    0, #first_step_range_start
                    5, #first_step_range_end
                    8, #rest_step_range_start
                    10, #rest_step_range_end
                    20.0, #noise_regularization_lambda_ac
                    0.055, #noise_regularization_lambda_kl
                    True, #noise_correction,
                    1, #fixed_point_iterations
                    2, #fixed_point_inversion_steps
                ],
            ]

            gr.Examples(examples=examples,
                        inputs=[
                            input_image,
                            src_prompt,
                            tgt_prompt,
                            edit_cfg,
                            seed,
                            number_of_renoising_iterations,
                            inersion_strength,
                            avg_gradients,
                            first_step_range_start,
                            first_step_range_end,
                            rest_step_range_start,
                            rest_step_range_end,
                            noise_regularization_lambda_ac,
                            noise_regularization_lambda_kl,
                            noise_correction,
                            fixed_point_iterations,
                            fixed_point_inversion_steps
                        ],
                        outputs=[
                            result
                        ],
                        fn=main_pipeline,
                        cache_examples=True)


    inputs = [
        input_image,
        src_prompt,
        tgt_prompt,
        edit_cfg,
        seed,
        number_of_renoising_iterations,
        inersion_strength,
        avg_gradients,
        first_step_range_start,
        first_step_range_end,
        rest_step_range_start,
        rest_step_range_end,
        noise_regularization_lambda_ac,
        noise_regularization_lambda_kl,
        noise_correction,
        fixed_point_iterations,
        fixed_point_inversion_steps
    ]
    outputs = [
        result
    ]
    run_button.click(fn=main_pipeline, inputs=inputs, outputs=outputs)

demo.queue(max_size=50).launch(share=False)