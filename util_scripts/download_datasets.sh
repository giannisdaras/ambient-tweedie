# Move to the folder you want to download your datasets
mkdir -p download_scripts
cd download_scripts

# Script that helps to download AFHQ, CelebA-HQ
wget https://github.com/clovaai/stargan-v2/raw/master/download.sh

# ------- CelebA-HQ --------- #
bash download.sh celeba-hq-dataset
mkdir -p celeba_hq_train_split
mv data/celeba_hq/train/male/* celeba_hq_train_split/
mv data/celeba_hq/train/female/* celeba_hq_train_split/
mkdir -p celeba_hq_eval_split
mv data/celeba_hq/val/male/* celeba_hq_eval_split/
mv data/celeba_hq/val/female/* celeba_hq_eval_split/
rm -rf data/



