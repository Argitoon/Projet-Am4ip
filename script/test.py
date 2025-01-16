import torch
import torch.nn.functional as F
import torch.nn as nn
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys

sys.path.append("/autofs/unitytravail/travail/ioyharcabal/M2S1/m2-iis-coco-et-ibaba/Am4ip/projet/src")

from am4ip.dataset import ProjectDataset
from am4ip.models import SimplifiedUnet, PSPNet
from am4ip.losses import DiceLoss
from am4ip.metrics import mIoU

from tools import create_transform, save_image, translate2color

from typing import Callable, Optional, Tuple, Union, Literal

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(model_type : Literal['unet', 'pspnet'], 
               checkpoint_path : str) -> Union[SimplifiedUnet, PSPNet]:
    """
    Load the model from the checkpoint file.
    
    :param model_type: The type of model to load (either 'unet' or 'pspnet').
    :param checkpoint_path: The path to the checkpoint file.
    :return: The loaded model.
    """
    
    if model_type == 'unet':
        model = SimplifiedUnet(34)  # ou Unet(34)
    elif model_type == 'pspnet':
        model = PSPNet(34)
    else:
        raise ValueError("Modèle invalide. Choisissez entre 'unet' ou 'pspnet'.")

    # Load state dict with appropriate map_location
    map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(checkpoint_path, map_location=map_location))
    model.to(map_location)
    model.eval()
    return model

def calculate_values(model, test_loader, loss, image_index : int = 0
                     ) -> Tuple[float, float, float]:
    """
    Calculate the accuracy and average loss and mIou score of the model on the test set.
    
    :param model: The model to evaluate.
    :param test_loader: The test set.
    :param loss: The loss function.
    :param image_index: The index of the image to save.
    :return: The accuracy, average loss and mIou score.
    """
    # Intit of the variables  
    total = 0
    correct = 0
    total_loss = 0
    total_miou = 0
    cpt = 0
    has_saved_images = False

    with torch.no_grad():
        for og_imgs, seg_imgs in tqdm(test_loader, desc="Testing"):
            og_imgs, seg_imgs = og_imgs.to(device), seg_imgs.to(device)
            outputs = model(og_imgs)
            
            if cpt == image_index and not has_saved_images:
                has_saved_images = True
                
                res_model_folder = "../results"
                if not os.path.exists(res_model_folder):
                    os.makedirs(res_model_folder)
                    
                res_model_folder = "../results/test-results"
                if not os.path.exists(res_model_folder):
                    os.makedirs(res_model_folder)
                    
                image = og_imgs[0].permute(1, 2, 0).cpu().numpy()
                path_name = res_model_folder + f"/{model_type}_{loss_type}_{dataset_type}_original.png"
                save_image(image, path_name)
                
                image = translate2color(seg_imgs[0].permute(1, 2, 0).cpu().detach().numpy())
                path_name = res_model_folder + f"/{model_type}_{loss_type}_{dataset_type}_target.png"
                save_image(image, path_name)
                
                out_imgs = torch.argmax(outputs, dim=1).unsqueeze(1)/255
                image = translate2color(out_imgs[0].permute(1, 2, 0).cpu().detach().numpy())
                path_name = res_model_folder + f"/{model_type}_{loss_type}_{dataset_type}_predicted.png"
                save_image(image, path_name)

            # Calculate the loss
            seg_imgs = (seg_imgs.squeeze(1) * 255).to(torch.long)
            loss_res = loss(outputs, seg_imgs)
            total_loss += loss_res.item()

            # Predictic the classes and calculate the accuracy
            probabilities = F.softmax(outputs, dim=1)
            _, predicted_classes = torch.max(probabilities, dim=1)

            correct += (predicted_classes == seg_imgs.squeeze(dim=1)).sum().item()
            total += seg_imgs.size(0) * seg_imgs.size(1) * seg_imgs.size(2)
            
            # Calculate MiOU
            miou_res = mIoU(predicted_classes, seg_imgs)
            total_miou = miou_res.item()
            cpt += 1  
            

    accuracy = correct / total
    avg_loss = total_loss / len(test_loader)
    avg_miou = total_miou / len(test_loader)
    return accuracy, avg_loss, avg_miou

if __name__ == '__main__':
    
    checkpoint_paths = []
    for dataset_type in ['denoised_rainy', 'sunny', 'rainy']:
        for model_type in ['unet', 'pspnet']:
            for loss_type in ['crossEntropy', 'dice']:
                checkpoint_paths.append(f"{model_type}_{loss_type}_{dataset_type}.pth")
    
    fraction = 0.5 # Fraction of the dataset to use
    
    path = './models/' # Path to the models
    
    def load_dataset(dataset_type: Literal['sunny', 'rainy', 'denoised_rainy'], 
                     fraction: float) -> ProjectDataset:
        """ 
        Load the dataset.
        
        :param dataset_type: The type of dataset to load (either 'sunny' or 'rainy').
        :param fraction: The fraction of the dataset to use.
        :return: The loaded dataset.
        """
        
        dataset = ProjectDataset(
            dataset_type = dataset_type,
            transform= create_transform(width=128, height=128),
            preprocess=True,
            fraction=fraction
        )
        return DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=4)
   
    # Calculate the accuracy and average loss for each model 
    #sunny_loader = load_dataset('sunny', fraction)
    previous_datatype = 'none'
    
    for checkpoint_path in checkpoint_paths:
        model_type = checkpoint_path.split('_')[0]
        loss_type = checkpoint_path.split('_')[1]
        if checkpoint_path.split('_')[2] == 'denoised' :
            dataset_type = 'denoised_rainy'
        else :
            dataset_type = checkpoint_path.split('_')[2].split('.')[0]
        
        # Load Dataset
        if dataset_type == 'sunny':
            if not previous_datatype == 'sunny': # Load the dataset if rainy is loaded
                previous_datatype = 'sunny'
                sunny_loader = load_dataset('sunny', fraction)
            dataset = sunny_loader
        elif dataset_type == 'rainy':
            if not previous_datatype == 'rainy': # Load the dataset if sunny is loaded
                previous_datatype = 'rainy'
                rainy_loader = load_dataset('rainy', fraction)
            dataset = rainy_loader
        elif dataset_type == 'denoised_rainy':
            if not previous_datatype == 'denoised_rainy' :
                previous_datatype = 'denoised_rainy'
                drainy_loader = load_dataset('denoised_rainy', fraction)
            dataset = drainy_loader
        else:
            raise ValueError("Invalid dataset type. Choose between 'sunny' or 'rainy'.")

        # Load model
        checkpoint_path = path + checkpoint_path
        model = load_model(model_type, checkpoint_path)
        
        # Load loss
        if loss_type == 'crossEntropy':
            criterion = nn.CrossEntropyLoss()
        elif loss_type == 'dice':
            criterion = DiceLoss()
        else:
            raise ValueError("Invalid loss type. Choose between 'crossEntropy' or 'dice'.")
        
        # Calculate accuracy, average loss and  
        
        accuracy, avg_loss, avg_miou = calculate_values(model, dataset, criterion)
        print(f"Model: {model_type} | {loss_type} | {dataset_type}")
        print(f"\tAccuracy: {accuracy}")
        print(f"\tAverage Loss: {avg_loss}")
        print(f"\tAverage Miou Score: {avg_miou}")