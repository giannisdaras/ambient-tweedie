"""Evaluates restoration performance of different models."""
import argparse
from models_catalog import models
from diffusers import UNet2DConditionModel
from diffusers import AutoencoderKL, DDPMScheduler
from diffusers import StableDiffusionXLPipeline
from diffusers.training_utils import EMAModel
import torch
import os
import ambient_utils
import matplotlib.pyplot as plt
from collections import defaultdict
import numpy as np
from ambient_utils.diffusers_utils import hot_sample
import wandb

parser = argparse.ArgumentParser(description="Evaluate denoisers.")
parser.add_argument("--dataset_path", type=str, help="Path to the dataset.", default="$LAION_RAW_DATA_RESCALED/")
parser.add_argument("--output_dir", type=str, help="Output directory.", default="$BASE_PATH/eval_denoisers/")
parser.add_argument("--resolution", type=int, help="Resolution of the dataset.", default=1024)
parser.add_argument("--num_eval_images", type=int, help="Number of images to evaluate.", default=4)
parser.add_argument("--eval_noise_levels", type=list, help="Noise level to evaluate.", default=[900, 800, 500, 100])
parser.add_argument("--prompt", type=str, help="Prompt to use for text generation.", default="face")
parser.add_argument("--whole_pipeline", action="store_true", help="Whether to load the whole pipeline or just the unet.")
parser.add_argument("--use_log_scale", action="store_true", help="Whether to visualize the results in log-scale.")

def main(args):
    wandb.init(
        project="ambient_diffusion",
        config=args,
        name="eval_denoisers")

    print("Loading dataset...")
    dataset_obj = ambient_utils.dataset_utils.ImageFolderDataset(path=args.dataset_path, use_labels=False, xflip=False, 
                                        corruption_probability=0.0, delta_probability=0.0, resolution=args.resolution, 
                                        max_size=args.num_eval_images)
    print("Loaded dataset.")

    os.makedirs(args.output_dir, exist_ok=True)
    results = {}
    for model_index, (model_key, model_dict) in enumerate(models.items()):
        print("Working with model: ", model_key)
        if model_dict['ckpt_path'] is not None:
            # get latest checkpoint
            checkpoint_numbers = [int(f.split('-')[-1]) for f in os.listdir(model_dict['ckpt_path']) if "checkpoint-" in f]
            max_number = str(sorted(checkpoint_numbers)[-1]).zfill(6)
            ckpt_path = os.path.join(model_dict['ckpt_path'], "checkpoint-" + max_number)
            print("Loading model from checkpoint: ", ckpt_path)
        else:
            ckpt_path = None
        pipe = ambient_utils.diffusers_utils.load_model(ckpt_path, vae_path="madebyollin/sdxl-vae-fp16-fix", sdxl_path="stabilityai/stable-diffusion-xl-base-1.0", 
                          trained_with_lora=model_dict['trained_with_lora']).to("cuda")
        latent_errors = defaultdict(list)
        errors = defaultdict(list)
        for dataset_item in dataset_obj:
            images = torch.tensor((2 * dataset_item['image'] - 1)[None], device="cuda").to(pipe.vae.dtype)
            images = images.repeat([len(args.eval_noise_levels), 1, 1, 1])
            with torch.no_grad():
                latent = pipe.vae.encode(images).latent_dist.sample()
            latent = latent * pipe.vae.config.scaling_factor
            steps = torch.tensor(args.eval_noise_levels)
            sigmas = ambient_utils.diffusers_utils.timesteps_to_sigma(steps.cpu(), pipe.scheduler.alphas_cumprod).cuda()
            noisy_latent = pipe.scheduler.add_noise(latent, torch.randn_like(latent), steps)
            with torch.no_grad():
                if not args.whole_pipeline:
                    noise_pred = ambient_utils.diffusers_utils.run_unet(pipe, noisy_latent, steps.to("cuda"), captions=[args.prompt] * latent.shape[0], return_noise=True).to(pipe.vae.dtype)
                    latent_x0_pred = ambient_utils.from_noise_pred_to_x0_pred_vp(noisy_latent, noise_pred, sigmas).to(pipe.vae.dtype)
                    x0_pred_mmse = pipe.vae.decode(latent_x0_pred / pipe.vae.config.scaling_factor).sample
                    ambient_utils.save_images(x0_pred_mmse, os.path.join(args.output_dir, f"{model_key}_mmse.png"), save_wandb=True)
                else:
                    x0_pred_mmse = hot_sample(pipe, noisy_latent, steps.to("cuda"), captions=[args.prompt] * latent.shape[0], stop_index=0)


            for index, step in enumerate(args.eval_noise_levels):
                if not args.whole_pipeline:
                    latent_errors[step].append(torch.nn.functional.mse_loss(latent_x0_pred[index], latent[index]).item())
                errors[step].append(torch.nn.functional.mse_loss(x0_pred_mmse[index], images[index]).item())

        results[model_key] = {}
        for step in args.eval_noise_levels: 
            if not args.whole_pipeline:
                latent_errors[step] = torch.tensor(latent_errors[step])
                results[model_key]["latent_mean_{}".format(step)] = latent_errors[step].mean().item()
                results[model_key]["latent_std_{}".format(step)] = latent_errors[step].std().item()
            errors[step] = torch.tensor(errors[step])
            results[model_key]["mean_{}".format(step)] = errors[step].mean().item()
            results[model_key]["std_{}".format(step)] = errors[step].std().item()
    
    ambient_utils.stylize_plots()
    # plot results
    figure = plt.figure()
    for model_key in results.keys():
        mse_numbers = [results[model_key]["mean_{}".format(step)] for step in args.eval_noise_levels]
        if args.use_log_scale:
            # make them log scale
            log_mse_numbers = [np.log10(mse) for mse in mse_numbers]
            plt.plot(args.eval_noise_levels, log_mse_numbers, label=model_key, marker='x')
        else:
            plt.plot(args.eval_noise_levels, mse_numbers, label=model_key, marker='x')

    plt.legend(results.keys())
    plt.savefig(os.path.join(args.output_dir, "results.pdf"))



if __name__ == "__main__":
    args = parser.parse_args()
    args = ambient_utils.expand_vars(args)
    main(args)
