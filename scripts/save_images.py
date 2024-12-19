import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
data_dir = os.path.join(project_root, 'results')

# Thêm đường dẫn vào sys.path nếu chưa có
if data_dir not in sys.path:
    sys.path.append(data_dir)

# Debug: Kiểm tra sys.path
print("Current sys.path:")
for p in sys.path:
    print(p)
import torch
import numpy as np
#from mpi4py import MPI
import socket
import argparse
import os
import json
import subprocess
from vdvae.hps import Hyperparams, parse_args_and_update_hparams, add_vae_arguments
from vdvae.utils import maybe_download
from vdvae.data import mkdir_p
from contextlib import contextmanager
import torch.distributed as dist
#from apex.optimizers import FusedAdam as AdamW
from vdvae.vae import VAE
from torch.nn.parallel.distributed import DistributedDataParallel
from vdvae.train_helpers import restore_params
from vdvae.image_utils import *
from vdvae.model_utils import *
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as T
import pickle

import argparse
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-sub", "--sub",help="Subject Number",default=1)
parser.add_argument("-bs", "--bs",help="Batch Size",default=30)
args = parser.parse_args()
sub=int(args.sub)
assert sub in [1,2,5,7]
batch_size=int(args.bs)

print('Libs imported')


  
class batch_generator_external_images(Dataset):

    def __init__(self, data_path):
        self.data_path = data_path
        self.im = np.load(data_path).astype(np.uint8)
        self.im = self.im[:len(self.im) // 10]


    def __getitem__(self,idx):
        img = Image.fromarray(self.im[idx])
        img = T.functional.resize(img,(64,64))
        img = torch.tensor(np.array(img)).float()
        #img = img/255
        #img = img*2 - 1
        return img

    def __len__(self):
        return  len(self.im)



image_path = 'data/processed_data/subj{:02d}/nsd_test_stim_sub{}.npy'.format(sub,sub)
test_images = batch_generator_external_images(data_path = image_path)
for idx in range(len(test_images)):  # Ensure we don't exceed the dataset size
    img_tensor = test_images[idx]  # Access the tensor

    # Convert tensor to a PIL Image
    img_array = img_tensor.numpy()  # Convert to NumPy array
    img_array = np.clip(img_array, 0, 255).astype(np.uint8)  # Ensure valid image values
    img = Image.fromarray(img_array)  # Convert to PIL Image
  # Access the image from the dataset  # Convert the numpy array to a PIL Image
    img.save(os.path.join('results/vdvae/origin-subj01/', f'test_image_{idx+1:03d}.png'))  # Save as PNG
    print(f"Saved image {idx+1} to /test_image_{idx+1:03d}.png")