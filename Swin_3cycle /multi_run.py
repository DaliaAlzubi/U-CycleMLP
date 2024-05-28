import os
from tqdm import tqdm
if __name__ == "__main__":
    conda_env_name = "main"

    for value in tqdm(range(40, 1001,10)):
        command = f"python test.py --dataset Synapse --cfg configs/swin_tiny_patch4_window7_224_lite.yaml  --volume_path Synapse --output_dir output  --base_lr 0.05 --img_size 224 --batch_size 24 --snap output/epoch_{value}.pth"
        print(command)
        # Uncomment the line below to execute the command
        os.system(command)