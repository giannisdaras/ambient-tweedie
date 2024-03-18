"""Visualize posterior samples and MMSE denoised images conditioned on noisy latents."""
from diffusers import DiffusionPipeline, AutoencoderKL
import torch
import numpy as np
import os
import ambient_utils
import argparse

parser = argparse.ArgumentParser(description="Generate images with a trained model.")
parser.add_argument("--image_path", type=str, help="Path to image for which we will visualize the noisy latents.", default='figures/example.jpeg')
parser.add_argument("--base_output_path", type=str, help="Base output path. Images will be saved in <base_output_path>/outputs/.", default="./")
parser.add_argument("--vae_path", type=str, help="Path to the trained model checkpoint.", default="madebyollin/sdxl-vae-fp16-fix")
parser.add_argument("--sdxl_path", type=str, help="Path to the trained model checkpoint.", default="stabilityai/stable-diffusion-xl-base-1.0")
parser.add_argument("--prompt", type=str, help="Prompt to use for text generation.", default="face")
parser.add_argument("--seed", type=int, help="Seed for generation.", default=42)

def main(args):
    os.makedirs(os.path.join(args.base_output_path, "outputs"), exist_ok=True)
    pipe = ambient_utils.diffusers_utils.load_model(None, args.vae_path, args.sdxl_path)
    pipe.to("cuda")


    example_image = ambient_utils.load_image(args.image_path)[:, :3].to(torch.float16) * 2 - 1
    steps = torch.tensor([500, 800]).cuda()

    sigmas = ambient_utils.diffusers_utils.timesteps_to_sigma(steps.cpu(), pipe.scheduler.alphas_cumprod)
    example_image = example_image.repeat([steps.shape[0], 1, 1, 1])

    latent = pipe.vae.encode(example_image).latent_dist.sample()
    resolution = example_image.shape[-2]
    del example_image  # free memory

    captions = [f"t={str(int(step))}" for step in steps]

    latent = latent * pipe.vae.config.scaling_factor
    noisy_latent = pipe.scheduler.add_noise(latent, torch.randn_like(latent), steps)
    ambient_utils.save_images(noisy_latent, os.path.join(args.base_output_path, "outputs/latents.png"), captions=captions, font_size=20, num_rows=1)

    # run full denoising
    print("Running full denoising.")
    with torch.no_grad():
        sampled_images = ambient_utils.diffusers_utils.hot_sample(pipe, noisy_latent, steps, captions=[""] * noisy_latent.shape[0], stop_index=0)
    
    print("Full denoising finished.")
    ambient_utils.save_images(sampled_images, os.path.join(args.base_output_path, "outputs/sampled.png"), captions=captions, font_size=60, num_rows=1)

    print("Running one-step denoising.")

    with torch.no_grad():
        denoised_latent = ambient_utils.diffusers_utils.run_unet(pipe, noisy_latent, steps).to(pipe.vae.dtype)
        ambient_utils.save_images(denoised_latent, os.path.join(args.base_output_path, "outputs/one_step_denoised_latents.png"))
        decoded = pipe.vae.decode(denoised_latent / pipe.vae.config.scaling_factor).sample
    ambient_utils.save_image(ambient_utils.tile_image(decoded, 1, noisy_latent.shape[0]), 
        os.path.join(args.base_output_path, "outputs/decoded_from_noisy.png"))
    print("Finished one-step denoising.")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)