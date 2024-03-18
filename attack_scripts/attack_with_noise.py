import argparse, os, glob
import torch
from PIL import Image
from tqdm import tqdm, trange
from torch import autocast

from contextlib import nullcontext

from transformers import AutoFeatureExtractor, logging
import pandas as pd
import ambient_utils
import wandb
from torchvision.ops import nms
import json
from diffusers import DiffusionPipeline, AutoencoderKL, DDPMScheduler
from diffusers.utils import load_image
from ambient_utils.diffusers_utils import hot_sample

logging.set_verbosity_error()

parser = argparse.ArgumentParser()
parser.add_argument(
    "--input_dir",
    type=str,
    help="Directory of images",
    default="$LAION_RAW_DATA/"
)
parser.add_argument(
    "--prompt",
    action='store_true',
    help="include caption from LAION"
)
parser.add_argument(
    "--outdir",
    type=str,
    nargs="?",
    help="dir to write results to",
    default="$BASE_PATH/xl_noise_attack/"
)
parser.add_argument(
    "--ddim_steps",
    type=int,
    default=20,
    help="number of ddim sampling steps",
)
parser.add_argument(
    "--ddim_eta",
    type=float,
    default=0.0,
    help="ddim eta (eta=0.0 corresponds to deterministic sampling",
)
parser.add_argument(
    "--n_iter",
    type=int,
    default=1,
    help="sample this often",
)
parser.add_argument(
    "--H",
    type=int,
    default=1024,
    help="image height, in pixel space",
)
parser.add_argument(
    "--W",
    type=int,
    default=1024,
    help="image width, in pixel space",
)
parser.add_argument(
    "--C",
    type=int,
    default=4,
    help="latent channels",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=1,
    help="how many samples to produce for each given prompt. A.k.a. batch size",
)
parser.add_argument(
    "--attempts", 
    type=int,
    default=1,
    help="Number of times to inpaint")
parser.add_argument(
    "--scale",
    type=float,
    default=7.5,
    help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
)
parser.add_argument(
    "--captions_loc",
    type=str,
    help="if specified, load prompts from this file",
    default="$CAPTIONS_LOC",
)

parser.add_argument(
    '--average_prompts',
    action='store_true',
    help='Whether to average the prompt embeddings. If not, only the first prompt is selected.',
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="the seed (for reproducible sampling)",
)
parser.add_argument(
    "--precision",
    type=str,
    help="evaluate at this precision",
    choices=["full", "autocast"],
    default="autocast"
)
parser.add_argument(
    "--timestep_nature",
    type=int,
    default=900,
    help="Controls the strength of the noise.",
)
parser.add_argument("--expr_id", type=str, default="xl_noise_attack", help="experiment id")
parser.add_argument("--whole_pipeline", action="store_true", help="whether to use the whole pipeline or just one step")

# custom models
parser.add_argument("--ckpt_path", type=str, default=None, help="Whether to load custom model.")
parser.add_argument("--default_prompt", type=str, default="", help="Default prompt to use if captions are not given.")


def load_captions(captions_loc):
    with open(opt.captions_loc) as f:
        captions = json.load(f)
    # replace key such that it only has the filename
    captions = {k.split("/")[-1]: v for k, v in captions.items()}
    return captions

def main(opt):
    assert opt.W == opt.H, "only square images supported."
    
    wandb.init(
        project="ambient_diffusion",
        config=opt,
        name=opt.expr_id)

    print("Using input directory: ", opt.input_dir)
    print("Using output directory: ", opt.outdir)
    
    if opt.captions_loc is not None:
        print("Using captions from: ", opt.captions_loc)
        captions = load_captions(opt.captions_loc)
        print("Loaded captions.")

        
    os.makedirs(opt.outdir, exist_ok=True)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    noise_scheduler = DDPMScheduler.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="scheduler")


    pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", vae=vae, 
        scheduler=noise_scheduler,
        torch_dtype=torch.float16, use_safetensors=True, variant="fp16").to(device)
    noise_scheduler = pipe.scheduler
    if opt.ckpt_path is not None:
        pipe.load_lora_weights(opt.ckpt_path)
    
    # Load configurations
    image_list = [x for x in glob.iglob(os.path.join(opt.input_dir, '*.png'))]
    image_list = sorted(image_list)
    for img_path in image_list:
        file_id = img_path.split("/")[-1]
        print(f"Processing {file_id}")
        file_id_no_ext = file_id.split(".")[0]
        # load image in [-1, 1]
        img = (2 * ambient_utils.load_image(img_path, resolution=opt.H) - 1).to(pipe.vae.dtype)

        latent = pipe.vae.encode(img).latent_dist.sample()
        latent = latent * pipe.vae.config.scaling_factor
        latent = latent.repeat(opt.attempts, 1, 1, 1)
        timesteps = torch.ones(latent.shape[0]).long().cuda() * opt.timestep_nature
        
        ambient_utils.save_images(img, os.path.join(opt.outdir, f"{file_id_no_ext}_input.png"))
        precision_scope = autocast if opt.precision=="autocast" else nullcontext
        with precision_scope("cuda"):
            if opt.captions_loc:
                prompt = captions[file_id][0]
            else:
                prompt = opt.default_prompt
            noisy_latent = noise_scheduler.add_noise(latent, torch.randn_like(latent), timesteps)
            with torch.no_grad():
                # compute MMSE denoiser
                latent_x0_pred = ambient_utils.diffusers_utils.run_unet(pipe, torch.clone(noisy_latent[0]).unsqueeze(0), timesteps[0].unsqueeze(0), captions=[prompt])
                x0_pred_mmse = torch.clone(pipe.vae.decode(latent_x0_pred / pipe.vae.config.scaling_factor).sample)
                if opt.whole_pipeline:
                        x0_pred = hot_sample(pipe, noisy_latent, timesteps, captions=[prompt] * noisy_latent.shape[0], stop_index=0)
            tiled_img = ambient_utils.tile_image(torch.cat([img, x0_pred_mmse, x0_pred], dim=0), n=1, m=2 + opt.attempts)
            for attempt_id, pred in enumerate(x0_pred):
                ambient_utils.save_image(pred, os.path.join(opt.outdir, f"{file_id_no_ext}_output_{attempt_id}.png"))
            ambient_utils.save_image(tiled_img, os.path.join(opt.outdir, f"{file_id_no_ext}_tiled.png"), save_wandb=True, wandb_down_factor=4)

if __name__ == "__main__":
    opt = parser.parse_args()
    opt = ambient_utils.expand_vars(opt)
    main(opt)
        
