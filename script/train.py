#source /net/ens/DeepLearning/python3/tensorflow2/bin/activate
#python -m ipykernel install --user --name=tensorflow2
#export PYTHONPATH=$PYTHONPATH:/autofs/unitytravail/travail/coseutin/m2-iis-coco-et-ibaba/Am4ip/projet/src

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import math
import re
import sys
import os

sys.path.append("/autofs/unitytravail/travail/coseutin/m2-iis-coco-et-ibaba/Am4ip/projet/src")
sys.path.append("/autofs/unitytravail/travail/ioyharcabal/M2S1/m2-iis-coco-et-ibaba/Am4ip/projet/src")

from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
from PIL import ImageEnhance
from collections import defaultdict

from am4ip.dataset import ProjectDataset
from am4ip.models import SimplifiedUnet
from am4ip.models import PSPNet
from am4ip.losses import DiceLoss
from am4ip.metrics import nMAE
from am4ip.metrics import mIoU

from typing import Literal

from tools import create_transform, split_dataset, translate2color, save_image, save_graphs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print(device)

def train_model(dataset_type : Literal["sunny", "rainy", "denoised_rainy"] = "sunny",
                model_type : Literal['unet', 'pspnet'] = 'unet',
                loss_type : Literal['crossEntropy', 'dice'] = 'crossEntropy',
                batch_size : int = 16, epochs : int = 50,
                step_save : int = 2, lr : float = 0.01,
                width : int = 128, height : int = 128,
                data_fraction : float = 0.075,
                it : int = 5,
                ) -> None :
    if loss_type == 'crossEntropy':
        criterion = nn.CrossEntropyLoss()
    elif loss_type == 'dice':
        criterion = DiceLoss()
    else:
        raise ValueError("Invalid loss type. Choose between 'crossEntropy' or 'dice'.")
    
    # Init tables which will be saved
    mean_train_loss_saved = np.zeros(math.ceil(epochs / step_save))
    mean_test_loss_saved = np.zeros(math.ceil(epochs / step_save))

    mean_train_accuracy_saved = np.zeros(math.ceil(epochs / step_save))
    mean_test_accuracy_saved = np.zeros(math.ceil(epochs / step_save))

    mean_train_mIoU_saved = np.zeros(math.ceil(epochs / step_save))
    mean_test_mIoU_saved = np.zeros(math.ceil(epochs / step_save))

    # Dataset 
    dataset = ProjectDataset(
        dataset_type = dataset_type,
        transform=create_transform(width, height),
        preprocess=True,
        fraction=data_fraction
    )
    
    # Split Dataset into train & test
    train_dataset, test_dataset = split_dataset(dataset)

    print(f"Nombre d'images dans train: {len(train_dataset)}")
    print(f"Nombre d'images dans test: {len(test_dataset)}")

    for i in range(it):
        print(f"Itération n°{i+1}/{it}\n")
        # Init tables which will be saved
        train_loss_saved = np.zeros(math.ceil(epochs / step_save))
        test_loss_saved = np.zeros(math.ceil(epochs / step_save))

        train_accuracy_saved = np.zeros(math.ceil(epochs / step_save))
        test_accuracy_saved = np.zeros(math.ceil(epochs / step_save))

        train_mIoU_saved = np.zeros(math.ceil(epochs / step_save))
        test_mIoU_saved = np.zeros(math.ceil(epochs / step_save))

        # Load Datasets
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

        # Model
        if model_type == 'unet' :
            model = SimplifiedUnet(34) #Unet(34)
        elif model_type == 'pspnet' :
            model = PSPNet(34)
        else :
            raise ValueError("Invalid model type. Choose between 'unet' or 'pspnet'.")
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        cpt = 0
        has_saved_images = False
        
        def save_debug_images(og_imgs : torch.Tensor, 
                              seg_imgs : torch.Tensor, 
                              outputs : torch.Tensor) -> None:
            """ 
            Save the last image of the train for debug
            
            :param og_imgs: Original images
            :param seg_imgs: Segmented images
            :param outputs: Model outputs
            
            :return: None
            """
            
            # Create the folder if it doesn't exist
            res_model_folder = "../results"
            if not os.path.exists(res_model_folder):
                os.makedirs(res_model_folder)
            
            res_model_folder = "../results/train-results"
            if not os.path.exists(res_model_folder):
                os.makedirs(res_model_folder)
            
            # Original Image
            image = og_imgs[0].permute(1, 2, 0).cpu().detach().numpy()
            path_name = res_model_folder + f"/{model_type}_{loss_type}Loss_{dataset_type}Data_imageExemple.png"
            save_image(image, path_name)
            
            # Target Image
            image = translate2color(seg_imgs[0].permute(1, 2, 0).cpu().detach().numpy())
            path_name = res_model_folder + f"/{model_type}_{loss_type}Loss_{dataset_type}Data_targetResult.png"
            save_image(image, path_name)
            
            # Pred Image
            out_imgs = torch.argmax(outputs, dim=1).unsqueeze(1)/255
            image = translate2color(out_imgs[0].permute(1, 2, 0).cpu().detach().numpy())
            path_name = res_model_folder + f"/{model_type}_{loss_type}Loss_{dataset_type}Data_predResult.png"
            save_image(image, path_name)
                
        # Train the model
        for epoch in range(epochs):
            model.train()
            with tqdm(train_loader) as tepoch:
                for og_imgs, seg_imgs in tepoch:
                    og_imgs, seg_imgs = og_imgs.to(device), seg_imgs.to(device)
                    optimizer.zero_grad()
                    outputs = model(og_imgs)

                    # Save the last image for debug
                    if epoch == epochs-1 and not has_saved_images:
                        save_debug_images(og_imgs, seg_imgs, outputs)
                        has_saved_images = True
                        
                    seg_imgs = (seg_imgs.squeeze(1)*255).to(torch.long)
                    loss = criterion(outputs, seg_imgs)
                    loss.backward()
                    optimizer.step()
        
        # Eval the model
            if epoch % step_save == 0:
                model.eval()
                with torch.no_grad():
                    cpt2 = 0
                    total = 0
                    for og_imgs, seg_imgs in train_loader:
                        og_imgs, seg_imgs = og_imgs.to(device), seg_imgs.to(device)

                        outputs = model(og_imgs)
                        seg_imgs = (seg_imgs.squeeze(1)*255).to(torch.long)
                        loss = criterion(outputs, seg_imgs)
                        
                        probabilities = F.softmax(outputs, dim=1)
                        max_probs, predicted_classes = torch.max(probabilities, dim=1)

                        train_accuracy_saved[cpt] += (predicted_classes == seg_imgs.squeeze(dim=1)).sum().item()
                        train_loss_saved[cpt] += loss.item()
                        train_mIoU_saved[cpt] += mIoU(predicted_classes, seg_imgs)

                        cpt2 += 1
                        total += seg_imgs.size(0)

                    train_loss_saved[cpt] /= cpt2
                    train_accuracy_saved[cpt] /= (total*width*height)
                    train_mIoU_saved[cpt] /= cpt2

                    cpt2 = 0
                    total = 0
                    for og_imgs, seg_imgs in test_loader:
                        og_imgs, seg_imgs = og_imgs.to(device), seg_imgs.to(device)

                        outputs = model(og_imgs)
                        seg_imgs = (seg_imgs.squeeze(1)*255).to(torch.long)
                        loss = criterion(outputs, seg_imgs)
                        
                        probabilities = F.softmax(outputs, dim=1)
                        max_probs, predicted_classes = torch.max(probabilities, dim=1)

                        test_accuracy_saved[cpt] += (predicted_classes == seg_imgs.squeeze(dim=1)).sum().item()
                        test_loss_saved[cpt] += loss.item()
                        test_mIoU_saved[cpt] += mIoU(predicted_classes, seg_imgs)

                        cpt2 += 1
                        total += seg_imgs.size(0)

                    test_loss_saved[cpt] /= cpt2
                    test_accuracy_saved[cpt] /= (total*width*height)
                    test_mIoU_saved[cpt] /= cpt2

                    cpt2 = 0

                cpt += 1
            print(f"Epoch {epoch+1}/{epochs}, Loss on train: {train_loss_saved[cpt-1]}, Loss on test: {test_loss_saved[cpt-1]}")

        mean_train_loss_saved += train_loss_saved
        mean_test_loss_saved += test_loss_saved

        mean_train_accuracy_saved += train_accuracy_saved
        mean_test_accuracy_saved += test_accuracy_saved

        mean_train_mIoU_saved += train_mIoU_saved
        mean_test_mIoU_saved += test_mIoU_saved
        
    # Save the model
    torch.save(model.state_dict(), f'{model_type}_{loss_type}_{dataset_type}.pth')
    print(f"Model {model_type} successfully saved !")

    mean_train_loss_saved = [x / it for x in mean_train_loss_saved]
    mean_test_loss_saved = [x / it for x in mean_test_loss_saved]

    mean_train_accuracy_saved = [x / it for x in mean_train_accuracy_saved]
    mean_test_accuracy_saved = [x / it for x in mean_test_accuracy_saved]

    mean_train_mIoU_saved = [x / it for x in mean_train_mIoU_saved]
    mean_test_mIoU_saved = [x / it for x in mean_test_mIoU_saved]

    # Extract the data
    if device.type == "cuda" and torch.is_tensor(mean_train_loss_saved[0]):
        mean_train_loss_saved = [tensor.cpu().item() for tensor in mean_train_loss_saved]
        mean_test_loss_saved = [tensor.cpu().item() for tensor in mean_test_loss_saved]
    if device.type == "cuda" and torch.is_tensor(mean_train_mIoU_saved[0]):
        mean_train_mIoU_saved = [tensor.cpu().item() for tensor in mean_train_mIoU_saved]
        mean_test_mIoU_saved = [tensor.cpu().item() for tensor in mean_test_mIoU_saved]

    # Save the results of the model
    epoch_range = range(1,epochs+1,step_save)
    save_graphs(epoch_range = epoch_range, nb_epoch = cpt,
                loss = (mean_train_loss_saved, mean_test_loss_saved),
                accuracy = (mean_train_accuracy_saved, mean_test_accuracy_saved),
                mIoU = (mean_train_mIoU_saved, mean_test_mIoU_saved),
                model_type = model_type, loss_type = loss_type, 
                dataset_type = dataset_type)
    
    
    # Final accuracy test
    print("Final mean accuracy on test:", mean_test_accuracy_saved[cpt-1])

    return

############################################################################
#                                                                          #
#                                   MAIN                                   #
#                                                                          #
############################################################################

if __name__ == '__main__':
    """ UNet model (Sunny & Rainy) - CrossEntropy """
    #train_model(dataset_type='sunny', model_type='unet', loss_type='crossEntropy')
    #train_model(dataset_type='rainy', model_type='unet', loss_type='crossEntropy')
    train_model(dataset_type='denoised_rainy', model_type='unet', loss_type='crossEntropy', data_fraction=0.2)
    
    """ PSPNet model (Sunny & Rainy) - CrossEntropy """
    #train_model(dataset_type='sunny', model_type='pspnet', loss_type='crossEntropy')
    #train_model(dataset_type='rainy', model_type='pspnet', loss_type='crossEntropy')
    train_model(dataset_type='denoised_rainy', model_type='pspnet', loss_type='crossEntropy', data_fraction=0.2)

    """ UNet model (Sunny & Rainy) - dice """
    #train_model(dataset_type='sunny', model_type='unet', loss_type='dice')
    #train_model(dataset_type='rainy', model_type='unet', loss_type='dice')
    train_model(dataset_type='denoised_rainy', model_type='unet', loss_type='dice', data_fraction=0.2)
    
    """ PSPNet model (Sunny & Rainy) - dice """
    #train_model(dataset_type='sunny', model_type='pspnet', loss_type='dice')
    #train_model(dataset_type='rainy', model_type='pspnet', loss_type='dice')
    train_model(dataset_type='denoised_rainy', model_type='pspnet', loss_type='dice', data_fraction=0.2)