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
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image

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
    default="$BASE_PATH/xl_masking_attack/"
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
    "--survival_probability",
    type=float,
    default=0.9,
    help="Controls the size of the mask.",
)

parser.add_argument(
    "--mask_with_yolo",
    action='store_true',
    help='Whether to use yolo to mask the image.',
)
parser.add_argument("--expr_id", type=str, default="xl_masking_attack", help="experiment id")

def load_captions(captions_loc):
    with open(opt.captions_loc) as f:
        captions = json.load(f)
    # replace key such that it only has the filename
    captions = {k.split("/")[-1]: v for k, v in captions.items()}
    return captions


def interpret_yolo_preds(yolo_preds, threshold=0.5):
    # Assuming yolo_preds is your tensor
    bbox = yolo_preds[0, :, :4]  # Extract bounding box coordinates
    conf = yolo_preds[0, :, 4]   # Extract objectness score
    class_probs = yolo_preds[0, :, 5:]  # Extract class probabilities

    # Get indices of boxes that have confidence scores above a certain threshold
    conf_threshold = 0.5
    indices = torch.where(conf > conf_threshold)

    # Filter bounding boxes and scores based on confidence threshold
    filtered_boxes = bbox[indices]
    filtered_conf = conf[indices]
    filtered_class_probs = class_probs[indices]

    # Apply non-maximum suppression
    keep = nms(filtered_boxes, filtered_conf, iou_threshold=threshold)

    # Final filtered and non-maximum suppressed results
    final_boxes = filtered_boxes[keep]
    final_conf = filtered_conf[keep]
    final_class_probs = filtered_class_probs[keep]
    return final_boxes, final_conf, final_class_probs

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

    if opt.mask_with_yolo:
        yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).cuda()

    start_code = None
    
        
    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    pipe = AutoPipelineForInpainting.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16, variant="fp16").to(device)    
            
        
    # Load configurations
    image_list = [x for x in glob.iglob(os.path.join(opt.input_dir, '*.png'))]
    image_list = sorted(image_list)
    for img_path in image_list:
        file_id = img_path.split("/")[-1]
        print(f"Processing {file_id}")
        file_id_no_ext = file_id.split(".")[0]
        img = 2 * ambient_utils.load_image(img_path, resolution=opt.H) - 1
                
        if opt.mask_with_yolo:
            yolo_preds = yolo_model(img)
            final_boxes, final_conf, final_class_probs = interpret_yolo_preds(yolo_preds)
            if len(final_boxes) == 0:
                print(f"No boxes found for image: {file_id}. Skipping to next file...")
                continue
            box_mask = torch.ones_like(img)
            for box in final_boxes:
                # make mask zero inside the box. box has format (x, y, w, h) where x, y represent the center and w, h are the width and height
                x, y, w, h = box
                box_mask[:, :, int(y - h / 2):int(y + h / 2), int(x - w / 2):int(x + w / 2)] = 0
            
            # if box_mask occupies more than 50% of the image, then discard
            if box_mask.sum() < 0.5 * box_mask.numel():
                print(f"Discarding image {file_id} as box mask is too big.")
                continue
             
             # if box_mask occupies less than 1% of the image, then discard
            if box_mask.sum() > 0.99 * box_mask.numel():
                print(f"Discarding image {file_id} as box mask is too small.")
                continue

            masked_image = img * box_mask
        else:
            box_mask = ambient_utils.get_box_mask_that_fits((1, 3, opt.H, opt.W), 
                survival_probability=opt.survival_probability, device=device)
            masked_image = img * box_mask
        ambient_utils.save_images(img, os.path.join(opt.outdir, f"{file_id_no_ext}_input.png"))
        ambient_utils.save_images(masked_image, os.path.join(opt.outdir, f"{file_id_no_ext}_masked.png"))
        box_mask_color_shifted = torch.where(box_mask == 0, torch.ones_like(box_mask), torch.ones_like(box_mask) * -1)
        ambient_utils.save_images(box_mask_color_shifted, os.path.join(opt.outdir, f"{file_id_no_ext}_mask.png"))
        precision_scope = autocast if opt.precision=="autocast" else nullcontext
        with precision_scope("cuda"):
            for attempt_id in trange(opt.attempts, desc="Sampling"):
                print(f"Running attempt {attempt_id} on {file_id}")
                if opt.captions_loc:
                    prompt = captions[file_id][0]
                else:
                    prompt = ""
                
                mask_pill = load_image(os.path.join(opt.outdir, f"{file_id_no_ext}_mask.png"))
                masked_image_pill = load_image(os.path.join(opt.outdir, f"{file_id_no_ext}_masked.png"))
                with torch.no_grad():
                    inpainted = pipe(prompt=prompt, image=masked_image_pill, mask_image=mask_pill, guidance_scale=8.0, num_inference_steps=opt.ddim_steps, strength=0.99).images[0]
                inpainted = ambient_utils.load_image(inpainted) * 2 - 1
                ambient_utils.save_images(inpainted, os.path.join(opt.outdir, f"{file_id_no_ext}_output_{attempt_id}.png"))
                tiled_img = ambient_utils.tile_image(torch.cat([img, masked_image, inpainted], dim=0), n=1, m=3)
                ambient_utils.save_image(tiled_img, os.path.join(opt.outdir, f"{file_id_no_ext}_tiled_{attempt_id}.png"), save_wandb=True, 
                    wandb_down_factor=4)


if __name__ == "__main__":
    opt = parser.parse_args()
    opt = ambient_utils.expand_vars(opt)
    main(opt)
        
