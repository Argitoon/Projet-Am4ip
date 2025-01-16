import torch
import numpy as np
import matplotlib.pyplot as plt
import re
import os
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
from PIL import ImageEnhance
from collections import defaultdict

from typing import Literal
from am4ip.dataset import ProjectDataset

class EnhanceTransform:
    def __init__(self, sharpness_factor: float = 1.5, contrast_factor: float = 1.5):
        """
        :param sharpness_factor: Factor to enhance sharpness (1.0 = no change).
        :param contrast_factor: Factor to enhance contrast (1.0 = no change).
        """
        self.sharpness_factor = sharpness_factor
        self.contrast_factor = contrast_factor

    def __call__(self, img: Image.Image) -> Image.Image:
        """
        Apply sharpness and contrast enhancement to the image.

        :param img: PIL.Image.Image
            Input image to transform.
        :return: PIL.Image.Image
            Transformed image.
        """
        if img.mode == 'RGB':
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(self.sharpness_factor)
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(self.contrast_factor)
        return img

def create_transform(width: int = 256, height: int = 256):
    """
    Create a composition of transformations to apply on the images.

    :param width: Width of the image after resizing.
    :param height: Height of the image after resizing.
    :return: A composition of transformations.
    """
    return transforms.Compose([
        EnhanceTransform(sharpness_factor=1.5, contrast_factor=1.5),
        transforms.Resize((width, height)),
        transforms.ToTensor()
    ])

def translate2color(values):
    """
    Convert a matrix of values to an RGB image using a predefined color mapping.

    :param values: numpy.ndarray
        2D array of float or int, normalized between 0-1 or containing discrete values.
    :return: numpy.ndarray
        3D array (H, W, 3) representing the RGB image with integer values [0-255].
    """
    values = np.round(values * 255).astype(int)
    colored_image = np.zeros((values.shape[0], values.shape[1], 3), dtype=int)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if values[i, j] == -1:  # Ground (Violet)
                colored_image[i, j] = [81, 0, 81]
            elif values[i, j] == 0:  # Road (Light Violet)
                colored_image[i, j] = [128, 64, 128]
            elif values[i, j] == 1:  # Sidewalk (Pink)
                colored_image[i, j] = [244, 35, 232]
            elif values[i, j] == 2:  # Building (Dark Gray)
                colored_image[i, j] = [70, 70, 70]
            elif values[i, j] == 3:  # Wall (Light Blue)
                colored_image[i, j] = [102, 102, 156]
            elif values[i, j] == 4:  # Fence (Light Brown)
                colored_image[i, j] = [190, 153, 153]
            elif values[i, j] == 5:  # Pole (Medium Gray)
                colored_image[i, j] = [153, 153, 153]  
            elif values[i, j] == 6:  # Traffic light (Orange)
                colored_image[i, j] = [250, 170, 30]
            elif values[i, j] == 7:  # Traffic sign (Yellow)
                colored_image[i, j] = [220, 220, 0]
            elif values[i, j] == 8:  # Vegetation (Light Green)
                colored_image[i, j] = [152, 251, 152]
            elif values[i, j] == 9:  # Terrain (Pale Green)
                colored_image[i, j] = [152, 251, 152]
            elif values[i, j] == 10:  # Sky (Blue)
                colored_image[i, j] = [70, 130, 180]
            elif values[i, j] == 11: # Person (Dark Red)
                colored_image[i, j] = [220, 20, 60]
            elif values[i, j] == 12:  # Rider (Bright Red)
                colored_image[i, j] = [255, 0, 0]
            elif values[i, j] == 13:  # Car (Dark Blue)
                colored_image[i, j] = [0, 0, 142]
            elif values[i, j] == 14: # Truck (Navy Blue)
                colored_image[i, j] = [0, 0, 70]
            elif values[i, j] == 15 or values[i, j] == 16 : # Bus (Light Blue)
                colored_image[i, j] = [0, 60, 100]
            elif values[i, j] > 17:  # Other (White) 
                colored_image[i, j] = [255, 255, 255]
            else:  # All other values become black
                colored_image[i, j] = [0, 0, 0]
    return colored_image

def group_by_series(image_paths : list) -> dict:
    """ 
    Group the image paths by series based on the first significant digit.
    
    :param image_paths: The list of image paths.
    
    :return: A dictionary of series.
    """
    
    series_dict = defaultdict(list)
    for path in image_paths:
        filename = os.path.basename(path)
        # Extraire le premier chiffre significatif
        match = re.search(r'[1-9]', filename)
        if match:
            series_prefix = match.group()  # Le premier chiffre significatif
            series_dict[series_prefix].append(path)
    return series_dict

def split_series(series_dict : dict, test_ratio : float = 0.2) -> tuple :
    """
    Split the series into training and testing sets based on the given test ratio.
    
    :param series_dict: The dictionary of series.
    :param test_ratio: Proportion of data to use for testing (default is 0.2).
    
    :return: A tuple of lists (train_paths, test_paths).
    """
    
    train_paths = []
    test_paths = []
    for series, paths in series_dict.items():
        split_idx = int(len(paths) * (1 - test_ratio))
        train_paths.extend(paths[:split_idx])
        test_paths.extend(paths[split_idx:])
    return train_paths, test_paths

def split_dataset(dataset, test_ratio = 0.2) :
    """
    Splits the dataset into training and testing sets based on the given test ratio.

    The function groups the dataset by series and splits them accordingly. It creates
    two `ProjectDataset` objects, one for training and one for testing, with updated
    image paths.

    :param dataset: The dataset of type `ProjectDataset`.
    :param test_ratio: Proportion of data to use for testing (default is 0.2).

    :return: A tuple of `ProjectDataset` objects (train_dataset, test_dataset).
    """
    series_dict = group_by_series(dataset.og_image_paths)
    train_paths, test_paths = split_series(series_dict, test_ratio=0.2)

    train_dataset = ProjectDataset(
        dataset_type=dataset.data_type,
        transform=dataset.transform,
        preprocess=False
        )
    train_dataset.og_image_paths = train_paths
    train_dataset.seg_image_paths = [p.replace(f'{dataset.data_type}_images', f'{dataset.data_type}_sseg').replace('.jpg', '.png') for p in train_paths]
    train_dataset.N = len(train_paths)

    test_dataset = ProjectDataset(
        dataset_type=dataset.data_type,
        transform=dataset.transform,
        preprocess=False
        )
    test_dataset.og_image_paths = test_paths
    test_dataset.seg_image_paths = [p.replace(f'{dataset.data_type}_images', f'{dataset.data_type}_sseg').replace('.jpg', '.png') for p in test_paths]
    test_dataset.N = len(test_paths)

    return (train_dataset, test_dataset)

def save_image(image, path_name) -> None:
    """
    Save an image to the specified file path.

    :param image: The image to be saved (can be a NumPy array, PyTorch tensor, or PIL image).
    :param path_name: The file path to save the image (including the file name and extension).
    """
    plt.imshow(image)
    plt.axis('off')
    plt.savefig(path_name)
    plt.clf()
    print(f"Image saved at: \'{path_name}\'")
    return

def save_graphs(epoch_range : list = [], nb_epoch : int = 0,
                loss : tuple[list, list] = ([], []),
                accuracy : tuple[list, list] = ([], []),
                mIoU : tuple[list, list] = ([], []),
                model_type : Literal['unet', 'pspnet'] = 'unet',
                loss_type : Literal['crossEntropy', 'dice'] = 'crossEntropy',
                dataset_type : Literal['sunny', 'rainy'] = 'sunny'
                ) -> None:
    
    """
    Save the training and test graphs for the model.
    
    :param epoch_range: The range of epochs.
    :param loss: The tuple of training and test loss.
    :param accuracy: The tuple of training and test accuracy.
    :param mIoU: The tuple of training and test mIoU.
    :param nb_epoch: The number of epochs.
    """
    
    # Create the results folder if it does not exist
    res_model_folder = "../results"
    if not os.path.exists(res_model_folder):
        os.makedirs(res_model_folder)
    
    res_model_folder = "../results/graphs"
    if not os.path.exists(res_model_folder):
        os.makedirs(res_model_folder)
        
    def save_graph(data : tuple, data_type : Literal['loss', 'accuracy', 'mIoU']) -> None:
        """
        Save the graph for the given data type.
        
        :param data: The tuple of training and test data.
        :param data_type: The type of data.
        """
        plt.plot(epoch_range, data[0][0:nb_epoch], label='Train', color='blue', marker='x')
        plt.plot(epoch_range, data[1][0:nb_epoch], label='Test', color='red', marker='x')
        plt.xlabel('Epochs')
        plt.ylabel(data_type)
        plt.title(f'Training and test {data_type} per epoch for {dataset_type}')
        plt.legend()
        path_name = res_model_folder + f"/{model_type}_{loss_type}Loss_{dataset_type}Data_{data_type}Graph.png"
        plt.savefig(path_name)
        plt.close()
        print(f"Model training and test {data_type} per epoch saved at: \'{path_name}\'")
        
    # Save the training and test loss graph
    save_graph(loss, 'loss')
    
    # Save the training and test accuracy graph
    save_graph(accuracy, 'accuracy')
    
    # Save the training and test mIoU graph
    save_graph(mIoU, 'mIoU')
    return