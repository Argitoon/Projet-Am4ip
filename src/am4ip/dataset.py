import numpy as np
from glob import glob
from PIL import Image
from numpy.typing import NDArray
from torch.utils.data import Dataset
from typing import Callable, Optional, Tuple, Union, Literal
import os
import torch
from torchvision import transforms
from tqdm import tqdm
import shutil
from pathlib import Path

from am4ip.utils import expanded_join

class ProjectDataset(Dataset):
    """
    Class utility to load, pre-process, put in batch, and convert to PyTorch convention images from the 
    ProjectDataset dataset.
    """
    
    """ 
    DATASET DIRECTORIES
    
    Here are defines the paths to the dataset and the cache directory.
    For sunny and rainy images, the dataset are located in '/net/ens/am4ip/datasets/project-dataset/'
    if you are in the cremi. You can change 'root_path' to your own path if you are working 
    on your computer.
    
    For denoised_rainy images, the dataset has been build by denoising the rainy images. 
    The shown path could not be available for you. You can find this dataset in google drive :
    https://drive.google.com/drive/folders/1YjToQFgl5SGmRaPVbmUw9h-qaqLniiiH?usp=sharing
    
    Finally, the cache directory is used to save the preprocessed images. 
    You can change 'cache_path' to your own path if you want.
    """

    # CREMI
    root_path = "/net/ens/am4ip/datasets/project-dataset/"
    cache_path = "./cache/"   # Directory for preprocessed images
    denoised_image_path = "/net/cremi/ioyharcabal/espaces/travail/M2S1/rainy_images/"

    # PERSONAL COMPUTER
    #root_path = "C:\\Users\\coren\\Desktop\\project-dataset\\"
    #cache_path = os.path.join(root_path, "cache") 
    
    def __init__(self, dataset_type: Literal["sunny", "rainy", "denoised_rainy"] = "sunny",
                 transform: Optional[Callable] = None,
                 preprocess: bool = True,
                 fraction: float = 1.0) -> None:
        """
        Class initialization.

        :param type: The type of dataset (either "sunny" or "rainy"), "Sunny" by default.
        :param transform: A set of transformations to apply on data.
        :param preprocess: If True, preprocess all images and save them to the cache.
        :param fraction: Fraction of the dataset to use (0.0 < fraction <= 1.0).
        """
        
        self.data_type = dataset_type
        self.preprocess = preprocess

        if dataset_type == "sunny" :
            self.N = 3779 # N_sunny = 3779
        elif dataset_type == "rainy":
            self.N = 3642 # N_rainy = 3642
        else : # denoised_rainy
            self.N = 907 # N_denoised_rainy = 907

        if not (0.0 < fraction <= 1.0):
            raise ValueError("fraction must be a float between ]0.0,1.0]")

        if dataset_type == "denoised_rainy":
            self.og_image_paths = sorted(glob(expanded_join(self.denoised_image_path, "*")), key=str.lower)
            file_names = {Path(p).name for p in self.og_image_paths}
            self.seg_image_paths = sorted(
                [p for p in glob(expanded_join(self.root_path, "rainy_sseg/*")) if Path(p).name in file_names],
                key=str.lower
            )
        else:
            self.og_image_paths = sorted(glob(expanded_join(self.root_path, f"{dataset_type}_images/*")), key=str.lower)
            self.seg_image_paths = sorted(glob(expanded_join(self.root_path, f"{dataset_type}_sseg/*")), key=str.lower)
        
        total_size = len(self.og_image_paths)
        step = max(1, int(1 / fraction))
        indices = np.arange(0, total_size, step)

        self.og_image_paths = [self.og_image_paths[i] for i in indices]
        self.seg_image_paths = [self.seg_image_paths[i] for i in indices]
        self.N = len(self.og_image_paths)

        self.transform = transform
        self.preprocessed = os.path.exists(self.cache_path)

        if preprocess:
            if os.path.exists(self.cache_path):
                shutil.rmtree(self.cache_path)
            os.makedirs(self.cache_path, exist_ok=True)
            self._preprocess_all()
        else:
            self.preprocessed = os.path.exists(self.cache_path)

    def _preprocess_all(self):
        """
        Preprocess and save all images to the cache.
        """
        for idx, (og_path, seg_path) in tqdm(enumerate(zip(self.og_image_paths, self.seg_image_paths)), total=self.N, desc="Preprocessing images"):
            og_img = Image.open(og_path)
            seg_img = Image.open(seg_path)

            if self.transform is not None:
                og_img = self.transform(og_img)
                seg_img = self.transform(seg_img)

            torch.save(og_img, os.path.join(self.cache_path, f"og_{idx}.pt"))
            torch.save(seg_img, os.path.join(self.cache_path, f"seg_{idx}.pt"))

        self.preprocessed = True

    def __len__(self):
        """
        Dataset size.
        :return: Size of the dataset.
        """
        return self.N

    def __getitem__(self, index: Union[slice, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a pair of preprocessed images.
        """
        og_img = torch.load(os.path.join(self.cache_path, f"og_{index}.pt"), weights_only=True)
        seg_img = torch.load(os.path.join(self.cache_path, f"seg_{index}.pt"), weights_only=True)
        return og_img, seg_img

class NightDataset(Dataset):
    root_path = "C:\\Users\\coren\\Desktop\\project-dataset\\"
    cache_path = os.path.join(root_path, "cache_night")

    def __init__(self, transform: Optional[Callable] = None,
                 preprocess: bool = True,
                 fraction: float = 1.0) -> None:
        self.og_image_paths = sorted(glob(os.path.join(self.root_path, "night", "*.jpg")), key=str.lower)
        self.noisy_image_paths = sorted(glob(os.path.join(self.root_path, "night-noisy", "*.jpg")), key=str.lower)
        
        if not (0.0 < fraction <= 1.0):
            raise ValueError("fraction must be a float between ]0.0,1.0]")

        total_size = len(self.og_image_paths)
        step = max(1, int(1 / fraction))
        indices = np.arange(0, total_size, step)

        self.og_image_paths = [self.og_image_paths[i] for i in indices]
        self.noisy_image_paths = [self.noisy_image_paths[i] for i in indices]
        self.N = len(self.og_image_paths)

        self.transform = transform
        self.preprocessed = os.path.exists(self.cache_path)

        if preprocess:
            if os.path.exists(self.cache_path):
                shutil.rmtree(self.cache_path)
            os.makedirs(self.cache_path, exist_ok=True)
            self._preprocess_all()
        else:
            self.preprocessed = os.path.exists(self.cache_path)

    def _preprocess_all(self):
        for idx, (og_path, noisy_path) in tqdm(enumerate(zip(self.og_image_paths, self.noisy_image_paths)), total=self.N, desc="Preprocessing night images"):
            og_img = Image.open(og_path).convert("RGB")
            noisy_img = Image.open(noisy_path).convert("RGB")

            if self.transform is not None:
                og_img = self.transform(og_img)
                noisy_img = self.transform(noisy_img)

            torch.save(og_img, os.path.join(self.cache_path, f"og_{idx}.pt"))
            torch.save(noisy_img, os.path.join(self.cache_path, f"noisy_{idx}.pt"))

        self.preprocessed = True

    def __len__(self):
        return self.N

    def __getitem__(self, index: Union[slice, int]) -> Tuple[torch.Tensor, torch.Tensor]:
        og_img = torch.load(os.path.join(self.cache_path, f"og_{index}.pt"), weights_only=True)
        noisy_img = torch.load(os.path.join(self.cache_path, f"noisy_{index}.pt"), weights_only=True)
        return og_img, noisy_img

