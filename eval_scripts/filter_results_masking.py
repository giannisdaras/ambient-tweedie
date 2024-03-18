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
import seaborn as sns
import wandb

def get_patch_mask(mask, image_size, patch_size=14, keep_cls=False, device='cuda'):
    """Computes a boolean mask of patches that are overlapping with the pixel mask.
        Args:
            mask: (num_channels, height, width)
            image_size: (height, width)
            patch_size: size of the ViT patch
            keep_cls: whether to keep the CLS token
        Returns:
            patch_mask: (num_patches,) boolean mask of patches that are overlapping with the mask
    """
    height, width = image_size
    num_patches = (height // patch_size) * (width // patch_size)
    patch_mask = np.zeros(num_patches, dtype=bool)
    for i in range(height // patch_size):
        for j in range(width // patch_size):
            patch_mask[i * (width // patch_size) + j] = np.any(mask[:, i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size])
    # add one patch for the CLS token
    patch_mask = np.concatenate(([keep_cls], patch_mask))
    patch_mask = torch.tensor(patch_mask, device=device)
    return patch_mask


def extract_features(feature_extractor, images, masks=None, localized_features=True, return_device='cpu'):
    """Extracts features from images using a ViT feature extractor.
        Args:
            feature_extractor: ViT feature extractor
            images: (batch_size, num_channels, height, width) images
            localized_features: whether to use localized features
        Returns:
            local_features: (batch_size, feature_dim)
    """
    if localized_features:
        if masks is None:
            masks = np.ones(images.shape[1:])
        patch_mask = get_patch_mask(masks, (images.size(-2), images.size(-1)), device=return_device)
        output = feature_extractor(images, is_training=True)   
        prenorm_output = output['x_prenorm']
        return prenorm_output.to(return_device), patch_mask[None, :, None]
    else:
        local_features = feature_extractor(images)
        return local_features.to(return_device)

def img_number_to_dataset_filename(file_index, files_per_folder=1000):
    folder_index = str(file_index // files_per_folder).zfill(5)
    file_index = str(file_index).zfill(8)
    matched_filename = folder_index + "/" + "img" + file_index + ".png"
    return matched_filename

def img_number_to_mask_filename(file_index):
    return str(file_index).zfill(6) + "_mask.png"

def match_indices_between_datasets(dataset1, dataset2, indices, name_mapping_fn):
    """Matches indices between two datasets.
        Args:
            dataset1: first dataset
            dataset2: second dataset
            indices: indices of dataset1
        Returns:
            matched_indices: indices of dataset2
    """
    matched_indices = []
    
    for index in indices:
        filename = dataset1[index]['filename']
        file_index = int(filename.split("_")[0])
        matched_filename = name_mapping_fn(file_index)
        matched_indices.append(dataset2.get_by_filename(matched_filename)["raw_idx"])
    return matched_indices

@click.command()
@click.option('--input_dir', 'input_dir',  help='Location of the folder where the network outputs are stored', metavar='PATH|URL',   type=str, required=True)
@click.option('--output_dir', 'output_dir',  help='Location of the folder where the outputs should be stored', metavar='PATH|URL',   type=str, required=True)
@click.option('--features_path', help='Path to save/load dataset features from', metavar='PATH|URL', type=str, required=True)
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
@click.option('--localized_features', help='Whether to use localized features', metavar='BOOL', type=bool, default=True, show_default=True)
@click.option('--skip_calculation', help='Whether to skip the calculation of the products. If True, products will be loaded from disk.', metavar='BOOL', type=bool, default=False, show_default=False)
@click.option('--enable_wandb', help='Whether to enable wandb logging', metavar='BOOL', type=bool, default=True, show_default=False)
@click.option('--expr_id', help='Experiment id', type=str, default="noise_attack_filtering", show_default=False)

def main(input_dir, output_dir, features_path, data, max_size, cache, workers, batch, batch_gpu, 
         seed, normalize, min_mask_ratio, top_k, size_for_feature_extraction, localized_features, 
         skip_calculation, enable_wandb, expr_id):
    
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

    dataset_obj = ambient_utils.dataset_utils.ImageFolderDataset(path=data, use_labels=False, xflip=False, cache=cache, 
                                        corruption_probability=0.0, delta_probability=0.0, resolution=1024)
    dist.print0("Loading feature extractor...")
    feature_extractor = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14').to('cuda')
    feature_extractor = torch.nn.parallel.DistributedDataParallel(feature_extractor, device_ids=[torch.device('cuda')], broadcast_buffers=False)
    dist.print0("Feature extractor loaded...")
    if not is_file(features_path):
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
                images = images.to('cuda').to(torch.float32)
                if size_for_feature_extraction is not None:
                    images = torch.nn.functional.interpolate(images, size=(size_for_feature_extraction, size_for_feature_extraction), 
                                                            mode='bilinear', align_corners=False)
                else:
                    # pad image to be divisible by 14
                    images = pad_image(images)
                
                curr_features = extract_features(feature_extractor, images, localized_features=localized_features)[0]

                features.append(curr_features)
        features = np.concatenate(features)
        np.save(features_path, features)
    else:
        features = np.load(features_path)


    os.makedirs(output_dir, exist_ok=True)
    
    dist.print0("Loading dataset...")
    outputs_dataset_obj = ambient_utils.dataset_utils.ImageFolderDataset(path=input_dir, use_labels=False, xflip=False, 
            cache=cache, corruption_probability=0.0, delta_probability=0.0, resolution=1024, must_contain="output", max_size=max_size)
    masks_dataset_obj = ambient_utils.dataset_utils.ImageFolderDataset(path=input_dir, use_labels=False, xflip=False,
            cache=cache, corruption_probability=0.0, delta_probability=0.0, resolution=1024, must_contain="mask", must_not_contain="masked")
    dist.print0("Dataset loaded...")

    if not skip_calculation:
        dist.print0("Calculating products...")
        max_products = []
        softmax_products = []
        max_indices = []
        kept_indices = []
        with torch.no_grad():
            for iter_index, dataset_iter in enumerate(tqdm(outputs_dataset_obj)):
                images = dataset_iter['image']
                filename = dataset_iter['filename']

                masks = masks_dataset_obj.get_by_filename(filename.split('_')[0] + '_mask.png')['image']
                if masks.sum() < min_mask_ratio * masks.size:
                    continue

                kept_indices.append(iter_index)
                images = torch.tensor(images, device='cuda').to(torch.float32).unsqueeze(0)

                if size_for_feature_extraction is not None:
                    images = torch.nn.functional.interpolate(images, size=(size_for_feature_extraction, size_for_feature_extraction), 
                                                            mode='bilinear', align_corners=False)
                    masks = torch.nn.functional.interpolate(torch.tensor(masks).unsqueeze(0), size=(size_for_feature_extraction, size_for_feature_extraction),
                                                            mode='nearest').squeeze(0).numpy()
                else:
                    # pad image to be divisible by 14
                    images = pad_image(images)
                
                curr_features, patch_mask = extract_features(feature_extractor, images, masks, localized_features=localized_features)

                # aggregate features only in the masked region
                localized_curr_features = (curr_features * patch_mask).sum(dim=1)
                localized_dataset_features = (torch.tensor(features) * patch_mask).sum(dim=1)

                # normalize dataset features
                if normalize:
                    localized_curr_features = localized_curr_features / np.linalg.norm(localized_curr_features, axis=1, keepdims=True)
                    localized_dataset_features = localized_dataset_features / np.linalg.norm(localized_dataset_features, axis=1, keepdims=True)            

                products = (localized_curr_features.cuda() @ localized_dataset_features.T.cuda()).squeeze().cpu()

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
        
        with open(os.path.join(output_dir, 'kept_indices.pkl'), 'wb') as f:
            pickle.dump(kept_indices, f)
    else:
        with open(os.path.join(output_dir, 'max_products.pkl'), 'rb') as f:
            max_products = pickle.load(f)
        
        with open(os.path.join(output_dir, 'softmax_products.pkl'), 'rb') as f:
            softmax_products = pickle.load(f)
        
        with open(os.path.join(output_dir, 'max_indices.pkl'), 'rb') as f:
            max_indices = pickle.load(f)
        
        with open(os.path.join(output_dir, 'kept_indices.pkl'), 'rb') as f:
            kept_indices = pickle.load(f)

    sns.set(style="whitegrid")
    plt.figure()
    sns.histplot(max_products, kde=True, bins=100, alpha=0.5)
    plt.title("Distribution of Similarity Values", fontsize=18)
    plt.xlabel("Similarity Value", fontsize=14)
    plt.ylabel("Number of generated samples", fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "similiarities.pdf"))
    dist.print0(f"Saved histogram to {os.path.join(output_dir, 'similiarities.pdf')}")


    # deal with the case where max_size < top_k
    top_k = min(top_k, len(max_products))
    
    # find indices of top products
    top_indices_from_computed = np.array(max_products).argsort()[::-1]

    # make the indices relative to the generated dataset and keep only the top-k ones
    top_k_indices = [kept_indices[i] for i in top_indices_from_computed[:top_k]]
    
    # find images that correspond to the highest products
    top_images = [torch.tensor(outputs_dataset_obj[index]['image']) for index in top_k_indices]

    top_k_dataset_indices = match_indices_between_datasets(outputs_dataset_obj, dataset_obj, top_k_indices, 
                                                           img_number_to_dataset_filename)    
    # find dataset images that correspond to the highest products
    top_dataset_images = [torch.tensor(dataset_obj[index]['image']) for index in top_k_dataset_indices]
    
    top_k_mask_indices = match_indices_between_datasets(outputs_dataset_obj, masks_dataset_obj, top_k_indices, img_number_to_mask_filename)
    top_masks = [torch.tensor(masks_dataset_obj[index]['image']) for index in top_k_mask_indices]
    

    # reshape images to have the same resolution if they don't.
    min_resolution = min(top_images[0].shape[-2], top_dataset_images[0].shape[-2])
    
    top_images = torch.stack(top_images, dim=0)
    top_masks = torch.stack(top_masks, dim=0)
    top_dataset_images = torch.stack(top_dataset_images, dim=0)

    # resize images to have the same resolution
    top_masked_images = top_images * (1 - top_masks)

    top_images = torch.nn.functional.interpolate(top_images, size=(min_resolution, min_resolution), mode='bilinear', align_corners=False)    
    top_masked_images = torch.nn.functional.interpolate(top_masked_images, size=(min_resolution, min_resolution), mode='bilinear', align_corners=False)
    top_dataset_images = torch.nn.functional.interpolate(top_dataset_images, size=(min_resolution, min_resolution), mode='bilinear', align_corners=False)
    cat_images = torch.cat((top_dataset_images, top_masked_images, top_images), dim=0) * 2 - 1
    save_image(ambient_utils.tile_image(cat_images, n=3, m=top_k), os.path.join(output_dir, 'top_images.png'))

if __name__ == '__main__':
    main()
        