import dnnlib
from dnnlib.util import load_image, pad_image
import click
import torch
from torch_utils import misc
from torch_utils import distributed as dist
from tqdm import tqdm
from dnnlib.util import pad_image, is_file, save_image
import numpy as np
from tqdm import tqdm
import os
import pickle
import matplotlib.pyplot as plt
import ambient_utils
import wandb
import seaborn as sns
import json
import copy

def load_captions(captions_loc):
    with open(captions_loc) as f:
        captions = json.load(f)
    # replace key such that it only has the filename
    captions = {k.split("/")[-1]: v for k, v in captions.items()}
    return captions

@click.command()
@click.option('--input_dir', 'input_dir',  help='Location of the folder where the network outputs are stored', metavar='PATH|URL',   type=str, required=True)
@click.option('--output_dir', 'output_dir',  help='Location of the folder where the outputs should be stored', metavar='PATH|URL',   type=str, required=True)
@click.option('--features_path', help='Path to save/load dataset features from', metavar='PATH|URL', type=str, required=True)
@click.option('--captions_loc', help='Location of the captions file', metavar='PATH|URL', type=str, default=None, show_default=True)
@click.option('--use_cls', help='Whether to use CLS token', metavar='BOOL', type=bool, default=True, show_default=True)
@click.option('--data',          help='Path to the dataset', metavar='ZIP|DIR',                     type=str, required=True)
@click.option('--max_size', help='Limit training samples.', type=int, default=None, show_default=True)
@click.option('--cache',         help='Cache dataset in CPU memory', metavar='BOOL',                type=bool, default=True, show_default=True)
@click.option('--workers',       help='DataLoader worker processes', metavar='INT',                 type=click.IntRange(min=1), default=1, show_default=True)
@click.option('--batch',         help='Total batch size', metavar='INT',                            type=click.IntRange(min=1), default=4, show_default=True)
@click.option('--batch-gpu',     help='Limit batch size per GPU', metavar='INT',                    type=click.IntRange(min=1))
@click.option('--seed',          help='Random seed  [default: random]', metavar='INT',              type=int, default=42)
@click.option('--normalize', help='Whether to normalize feature vectors', metavar='BOOL', type=bool, default=True, show_default=True)
@click.option('--min_mask_ratio', help='Minimum ratio of masked pixels', metavar='FLOAT', type=float, default=0.1, show_default=True)
@click.option('--top_k', help='Number of top-k images to save', metavar='INT', type=int, default=10, show_default=True)
@click.option('--size_for_feature_extraction', help='Size of images for feature extraction', metavar='INT', type=int, default=224, show_default=True)
@click.option('--skip_calculation', help='Whether to skip the calculation of the products. If True, products will be loaded from disk.', metavar='BOOL', type=bool, default=False, show_default=False)
@click.option('--enable_wandb', help='Whether to enable wandb logging', metavar='BOOL', type=bool, default=True, show_default=False)
@click.option('--expr_id', help='Experiment id', type=str, default="noise_attack_filtering", show_default=False)
@click.option('--shift_range', help='Whether to shift images from [0, 1] to [-1, 1].', metavar='BOOL', type=bool, default=False, show_default=False)

def main(input_dir, output_dir, features_path, captions_loc, use_cls, data, max_size, cache, workers, batch, batch_gpu, 
         seed, normalize, min_mask_ratio, top_k, size_for_feature_extraction, 
         skip_calculation, enable_wandb, expr_id, shift_range):

    torch.multiprocessing.set_start_method('spawn')
    dist.init()


    if enable_wandb and dist.get_rank() == 0:
        wandb.init(project="ambient", name=expr_id)
        wandb.config.update(locals())


    if seed is None:
        seed = torch.randint(1 << 31, size=[], device=torch.device('cuda'))
        torch.distributed.broadcast(seed, src=0)
        seed = int(seed)
    # Select batch size per GPU.
    batch_gpu_total = batch // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    

    c = dnnlib.EasyDict()
    
    if not skip_calculation:
        dist.print0("Loading feature extractor...")
        feature_extractor = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14').to('cuda')
        feature_extractor = torch.nn.parallel.DistributedDataParallel(feature_extractor, device_ids=[torch.device('cuda')], broadcast_buffers=False)
        dist.print0("Feature extractor loaded...")

    if not is_file(features_path):
        dataset_obj = ambient_utils.dataset_utils.ImageFolderDataset(path=data, use_labels=False, xflip=False, cache=cache, 
                                            corruption_probability=0.0, delta_probability=0.0, resolution=1024)
        dist.print0("Computing dataset embeddings..")

        c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=workers, prefetch_factor=2)
        dataset_sampler = torch.utils.data.distributed.DistributedSampler(dataset_obj, num_replicas=dist.get_world_size(), rank=dist.get_rank(), seed=seed, shuffle=False)

        dataset_iterator = iter(
            torch.utils.data.DataLoader(
                dataset=dataset_obj, sampler=dataset_sampler, batch_size=batch_gpu, **c.data_loader_kwargs)
            )

        features = []
        with torch.no_grad():
            for dataset_iter in tqdm(dataset_iterator):
                images = dataset_iter['image']
                # images are in [0, 1]
                images = images.to('cuda').to(torch.float32)
                if shift_range:
                    images = 2 * images - 1
                if size_for_feature_extraction is not None:
                    images = torch.nn.functional.interpolate(images, size=(size_for_feature_extraction, size_for_feature_extraction), 
                                                            mode='bilinear', align_corners=False)
                else:
                    # pad image to be divisible by 14
                    images = pad_image(images)
                if use_cls:
                    curr_features = feature_extractor(images).cpu().numpy()
                else:
                    curr_features = feature_extractor(images, is_training=True)['x_norm_patchtokens'].sum(axis=1).cpu().numpy()
                features.append(curr_features)
        features = np.concatenate(features)
        np.save(features_path, features)
    else:
        features = np.load(features_path)


    os.makedirs(output_dir, exist_ok=True)

    flag_diff_names = False

    dist.print0("Loading dataset...")
    try:
        outputs_dataset_obj = ambient_utils.dataset_utils.ImageFolderDataset(path=input_dir, use_labels=False, xflip=False, 
                cache=cache, corruption_probability=0.0, delta_probability=0.0, resolution=1024, must_contain="output", max_size=max_size)
    except:
        # if the dataset is not in the format of the outputs, try to load it as a regular dataset
        outputs_dataset_obj = ambient_utils.dataset_utils.ImageFolderDataset(path=input_dir, use_labels=False, xflip=False, 
                cache=cache, corruption_probability=0.0, delta_probability=0.0, resolution=1024, max_size=max_size)
        flag_diff_names = True

    dist.print0("Dataset loaded...")

    if not skip_calculation:
        dist.print0("Calculating products...")
        max_products = []
        softmax_products = []
        max_indices = []
        with torch.no_grad():
            if normalize:
                features = features / np.linalg.norm(features, axis=1, keepdims=True)            
            for iter_index, dataset_iter in enumerate(tqdm(outputs_dataset_obj)):
                images = dataset_iter['image']
                images = torch.tensor(images, device='cuda').to(torch.float32).unsqueeze(0)
                if shift_range:
                    images = 2 * images - 1
                if size_for_feature_extraction is not None:
                    images = torch.nn.functional.interpolate(images, size=(size_for_feature_extraction, size_for_feature_extraction), 
                                                            mode='bilinear', align_corners=False)
                else:
                    # pad image to be divisible by 14
                    images = pad_image(images)

                if use_cls:                
                    curr_features = feature_extractor(images).cpu().numpy()
                else:
                    curr_features = feature_extractor(images, is_training=True)['x_norm_patchtokens'].sum(axis=1).cpu().numpy()
                # normalize dataset features
                if normalize:
                    curr_features = curr_features / np.linalg.norm(curr_features, axis=1, keepdims=True)

                products = torch.tensor((curr_features @ features.T).squeeze())
                # get normalized probabilities from logits
                softmax_products.append(torch.nn.functional.softmax(products).max())
                max_products.append(float(products.max()))
                max_indices.append(products.argmax())
                
        with open(os.path.join(output_dir, 'max_products.pkl'), 'wb') as f:
            pickle.dump(max_products, f)
        
        with open(os.path.join(output_dir, 'softmax_products.pkl'), 'wb') as f:
            pickle.dump(softmax_products, f)
        
        with open(os.path.join(output_dir, 'max_indices.pkl'), 'wb') as f:
            pickle.dump(max_indices, f)        
    else:
        with open(os.path.join(output_dir, 'max_products.pkl'), 'rb') as f:
            max_products = pickle.load(f)
        
        with open(os.path.join(output_dir, 'softmax_products.pkl'), 'rb') as f:
            softmax_products = pickle.load(f)
        
        with open(os.path.join(output_dir, 'max_indices.pkl'), 'rb') as f:
            max_indices = pickle.load(f)
    
    # create histogram of max products

    sns.set(style="whitegrid")
    plt.figure()
    sns.histplot(max_products, kde=True, bins=100, alpha=0.5)
    plt.title("Distribution of Similarity Values", fontsize=18)
    plt.xlabel("Similarity Value", fontsize=14)
    plt.ylabel("Number of generated samples", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "similiarities.pdf"))
    dist.print0(f"Saved histogram to {os.path.join(output_dir, 'similiarities.pdf')}")


    # compute some statistics
    # number of images with similarity > 0.9
    num_similar = sum([1 for x in max_products if x > 0.9])
    # number of images with similarity > 0.95
    num_very_similar = sum([1 for x in max_products if x > 0.95])
    # number of images with similarity > 0.99
    num_very_very_similar = sum([1 for x in max_products if x > 0.99])
    dist.print0("Number of images with similarity > 0.9: ", num_similar)
    dist.print0("Number of images with similarity > 0.95: ", num_very_similar)
    dist.print0("Number of images with similarity > 0.99: ", num_very_very_similar)
    # convert them to percentages
    num_similar = 100 * round(num_similar / len(max_products), 3)
    num_very_similar = 100 * round(num_very_similar / len(max_products), 3)
    num_very_very_similar = 100 * round(num_very_very_similar / len(max_products), 3)
    dist.print0("Percentage of images with similarity > 0.9: ", num_similar)
    dist.print0("Percentage of images with similarity > 0.95: ", num_very_similar)
    dist.print0("Percentage of images with similarity > 0.99: ", num_very_very_similar)

    # print average similarity and std
    dist.print0("Average similarity: ", np.mean(max_products))
    dist.print0("Std of similarity: ", np.std(max_products))

    # print max similarity
    dist.print0("Max similarity: ", np.max(max_products))
    
    # deal with the case where max_size < top_k
    top_k = min(top_k, len(max_products))

    # find indices of top products
    top_indices_from_computed = np.array(max_products).argsort()[::-1]
    dataset_items = [outputs_dataset_obj[idx] for idx in top_indices_from_computed[:top_k]]
    # get base filenames (they are supposed to look like 000000_output_{attempt_id}.png)
    base_filenames = [item['filename'].split('_')[0].replace(".png", "") for item in dataset_items]
    gen_filenames = [item['filename'] for item in dataset_items]
    dist.print0("Top filenames in generated dataset: ", base_filenames)



    if not flag_diff_names:
        # load tiled images
        top_mmse_images = [2 * ambient_utils.load_image(os.path.join(input_dir, base_filename + "_tiled.png"))[:, :, :, 1024:2048] - 1 for base_filename in base_filenames]
        top_mmse_images = torch.cat(top_mmse_images)


    # find corresponding indices in the dataset
    top_matched_indices = torch.tensor(max_indices)[top_indices_from_computed.tolist()][:top_k]
    base_filenames = [str(int(x)).zfill(5) for x in top_matched_indices]
    dist.print0("Top filenames in dataset: ", base_filenames)
    if captions_loc is not None:
        captions = load_captions(captions_loc)
        selected_captions = [captions[base_filename + ".png"][0] for base_filename in base_filenames]
    else:
        selected_captions = ["" for base_filename in base_filenames]

    state_dict = {
        "captions": selected_captions,
        "filenames": base_filenames,
        "gen_filenames": gen_filenames,
        "max_products": max_products,
    }
    with open(os.path.join(output_dir, "top_matches.json"), "w") as f:
        json.dump(state_dict, f)
    dist.print0(f"Saved top matches to json file {os.path.join(output_dir, 'top_matches.json')}")

    # load top dataset images
    top_images = [2 * ambient_utils.load_image(os.path.join(data, base_filename + ".png"), resolution=1024) - 1 for base_filename in base_filenames]
    top_images = torch.cat(top_images)
    # load top generated images
    top_generated_images = [2 * ambient_utils.load_image(os.path.join(input_dir, x['filename'])) - 1 for x in dataset_items]
    top_generated_images = torch.cat(top_generated_images)

    if not flag_diff_names:
        to_concat = [top_images, top_generated_images, top_mmse_images]
    else:
        to_concat = [top_images, top_generated_images]
    all_images = torch.cat(to_concat)
    
    ambient_utils.save_images(all_images, os.path.join(output_dir, "top_matches.png"), 
                              num_rows=len(to_concat), 
                              save_wandb=True, down_factor=8)


if __name__ == '__main__':
    main()
