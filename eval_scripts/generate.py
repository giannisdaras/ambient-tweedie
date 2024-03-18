"""Generate images using a pre-trained or finetuned SDXL model."""
import argparse
from diffusers.training_utils import EMAModel
from diffusers import UNet2DConditionModel
from diffusers import AutoencoderKL
from diffusers import StableDiffusionXLPipeline
import os
import torch
import ambient_utils
from models_catalog import models
from tqdm import tqdm
import warnings
import json

parser = argparse.ArgumentParser(description="Generate images with a trained model.")
# Checkpoint params
parser.add_argument("--model_key", type=str, help="Key of the model to use.", default=None)
parser.add_argument("--ckpt_path", type=str, help="Path to the trained model checkpoint.", default=None)
parser.add_argument("--trained_with_lora", action="store_true", help="Whether the model was trained with LoRA.")
parser.add_argument("--timestep_nature", type=int, help="For noisy ambient.", default=None)
parser.add_argument("--use_latest_checkpoint", action="store_true", help="Whether to use the latest checkpoint.")

# Sampling params
parser.add_argument("--num_images", type=int, help="Number of images to generate.", default=50000)
parser.add_argument("--batch_size", type=int, help="Batch size for generation.", default=16)
parser.add_argument("--num_inference_steps", type=int, help="Number of inference steps.", default=25)
parser.add_argument("--early_stop_generation", action="store_true", help="Whether to stop generation early.", default=False)
parser.add_argument("--captions_loc", type=str, help="If specified, prompts will be loaded from file.", default=None)



# Generic params
parser.add_argument("--base_output_path", type=str, help="Base output path.", default="$BASE_PATH")
parser.add_argument("--vae_path", type=str, help="Path to the trained model checkpoint.", default="madebyollin/sdxl-vae-fp16-fix")
parser.add_argument("--sdxl_path", type=str, help="Path to the trained model checkpoint.", default="stabilityai/stable-diffusion-xl-base-1.0")
parser.add_argument("--prompt", type=str, help="Prompt to use for text generation.", default="face")
parser.add_argument("--seed", type=int, help="Seed for generation.", default=42)


def load_captions(captions_loc):
    with open(captions_loc) as f:
        captions = json.load(f)
    # replace key such that it only has the filename
    captions = {k.split("/")[-1]: v for k, v in captions.items()}
    return captions



def main(args):
    args = ambient_utils.expand_vars(args)
    # create output directory, ok if it exists already
    os.makedirs(args.base_output_path, exist_ok=True)

    if args.captions_loc is not None:
        captions = list(load_captions(args.captions_loc).values())
    if args.model_key is not None:
        args.ckpt_path = models[args.model_key]["ckpt_path"]
        args.trained_with_lora = models[args.model_key]["trained_with_lora"]
        args.timestep_nature = models[args.model_key]["timestep_nature"]
    else:
        if args.ckpt_path is None:
            warnings.warn("No model key or checkpoint path provided, using SDXL model.")

    if args.use_latest_checkpoint and args.ckpt_path is not None:
        checkpoint_numbers = [int(f.split('-')[-1]) for f in os.listdir(args.ckpt_path) if "checkpoint-" in f]
        max_number = str(sorted(checkpoint_numbers)[-1]).zfill(6)
        args.ckpt_path = os.path.join(args.ckpt_path, "checkpoint-" + max_number)

    if args.ckpt_path is not None:
        ambient_utils.dist.print0(f"Working with checkpoint path: {args.ckpt_path}")
        full_output_path = os.path.join(args.base_output_path, args.ckpt_path.split("/")[-2] + args.ckpt_path.split("/")[-1] + f"_{args.num_inference_steps}_early_stop_{args.early_stop_generation}")
        os.makedirs(full_output_path, exist_ok=True)
    else:
        full_output_path = os.path.join(args.base_output_path, "sdxl")
        os.makedirs(full_output_path, exist_ok=True)

    ambient_utils.dist.print0(f"Saving to: {full_output_path}")
    pipe = ambient_utils.diffusers_utils.load_model(args.ckpt_path, args.vae_path, args.sdxl_path, args.trained_with_lora)

    pipe = pipe.to("cuda")
    if args.timestep_nature is not None and args.early_stop_generation:
        stop_index = args.timestep_nature
    else:
        stop_index = None

    generator = torch.Generator(device="cuda").manual_seed(args.seed + ambient_utils.dist.get_rank())
    for image_indices in tqdm(args.rank_batches, disable=ambient_utils.dist.get_rank() != 0):
        batch_size = len(image_indices)
        
        # check if image already exists, if so, skip to the next one
        image_name_to_check = os.path.join(full_output_path, str(int(image_indices[0])).zfill(6) + ".png")
        if os.path.exists(image_name_to_check):
            print(f"Path {image_name_to_check} exists, continuing..")
            continue
        if args.captions_loc is not None:
            # select at random from captions
            random_indices = torch.randint(0, len(captions), (batch_size,))
            prompts = [captions[x][0] for x in random_indices]
        else:
            prompts = [args.prompt] * args.batch_size
        with torch.no_grad():
            denoising_end = 1 - args.timestep_nature / pipe.scheduler.config.num_train_timesteps
            pipe_kwargs = {
                "generator": generator,
            }
            tensor_images = ambient_utils.diffusers_utils.sample_with_early_stop(pipe, denoising_end, prompts, num_inference_steps=args.num_inference_steps, **pipe_kwargs)

        for image_index, image in zip(image_indices, tensor_images):
            # make the image name to have 6 digits
            image_name = os.path.join(full_output_path, str(int(image_index)).zfill(6) + ".png")
            ambient_utils.save_image(image, image_name)

if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    ambient_utils.dist.init()
    args = parser.parse_args()

    num_batches = ((args.num_images - 1) // (args.batch_size * ambient_utils.dist.get_world_size()) + 1) * ambient_utils.dist.get_world_size()
    seeds = torch.arange(num_batches * args.batch_size)
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[ambient_utils.dist.get_rank() :: ambient_utils.dist.get_world_size()]
    batches_per_process = len(rank_batches)
    ambient_utils.dist.print0(f"Each process will get {len(rank_batches)} batches.")
    args.rank_batches = rank_batches    
    main(args)
