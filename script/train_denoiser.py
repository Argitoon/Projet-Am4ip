#source /net/ens/DeepLearning/python3/tensorflow2/bin/activate
#python -m ipykernel install --user --name=tensorflow2
#export PYTHONPATH=$PYTHONPATH:/autofs/unitytravail/travail/coseutin/m2-iis-coco-et-ibaba/Am4ip/projet/src

import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import math
import sys
import os

#sys.path.append("/autofs/unitytravail/travail/coseutin/m2-iis-coco-et-ibaba/Am4ip/projet/src")
sys.path.append("/autofs/unitytravail/travail/ioyharcabal/M2S1/m2-iis-coco-et-ibaba/Am4ip/projet/src")
#sys.path.append(os.path.abspath("C:/Users/coren/Desktop/m2-iis-coco-et-ibaba/Am4ip/projet/src"))

from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from am4ip.dataset import NightDataset
from am4ip.models import DenoiserUnet
from am4ip.losses import DiceLoss

from typing import Callable, Optional, Tuple, Union, Literal

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print(device)

def split_dataset(dataset, test_ratio=0.2):
    total_size = len(dataset.og_image_paths)
    test_size = int(total_size * test_ratio)
    train_size = total_size - test_size

    train_og_paths = dataset.og_image_paths[:train_size]
    test_og_paths = dataset.og_image_paths[train_size:]

    train_noisy_paths = dataset.noisy_image_paths[:train_size]
    test_noisy_paths = dataset.noisy_image_paths[train_size:]

    train_dataset = NightDataset(
        transform=dataset.transform,
        preprocess=False
    )
    train_dataset.og_image_paths = train_og_paths
    train_dataset.noisy_image_paths = train_noisy_paths
    train_dataset.N = len(train_og_paths)

    test_dataset = NightDataset(
        transform=dataset.transform,
        preprocess=False
    )
    test_dataset.og_image_paths = test_og_paths
    test_dataset.noisy_image_paths = test_noisy_paths
    test_dataset.N = len(test_og_paths)

    return train_dataset, test_dataset


def save_image(image, path_name) -> None:
    plt.imshow(image)
    plt.axis('off')
    plt.savefig(path_name)
    plt.clf()
    print(f"Image saved at: \'{path_name}\'")
    return

############################################################################
#                                                                          #
#                                  TRAIN                                   #
#                                                                          #
############################################################################

def train_model(
                loss_type : Literal['crossEntropy', 'dice', 'MSE'] = 'crossEntropy',
                batch_size : int = 16, epochs : int = 151,
                step_save : int = 10, lr : float = 0.001,
                width : int = 128, height : int = 128,
                data_fraction : float = 0.5
                ) -> None :
    model = DenoiserUnet()
    model.to(device)
    
    if loss_type == 'crossEntropy':
        criterion = nn.CrossEntropyLoss()
    elif loss_type == 'dice':
        criterion = DiceLoss()
    elif loss_type == 'MSE':
        criterion = nn.MSELoss()
    else:
        raise ValueError("Invalid loss type. Choose between 'crossEntropy', 'dice', or 'miou'.")
    
    # Dataset 
    dataset = NightDataset(
        transform=transforms.Compose([
            transforms.Resize((width, height)),
            transforms.ToTensor()
        ]),
        preprocess=True,
        fraction=data_fraction
    )
    
    # Split Dataset into train & test
    train_dataset, test_dataset = split_dataset(dataset)

    print(f"Nombre d'images dans train: {len(train_dataset)}")
    print(f"Nombre d'images dans test: {len(test_dataset)}")
    
    # Load Datasets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Init Loss and Accuracy tables
    train_loss_saved = [0]*math.ceil(epochs / step_save)
    test_loss_saved = [0]*math.ceil(epochs / step_save)

    train_accuracy_saved = [0]*math.ceil(epochs / step_save)
    test_accuracy_saved = [0]*math.ceil(epochs / step_save)
    cpt = 0
    has_saved_images = False
        
    def save_accuracy_loss_results(epochs, train_loss, test_loss, train_accuracy, test_accuracy, cpt) -> None:
        res_model_folder = "../results"
        if not os.path.exists(res_model_folder):
            os.makedirs(res_model_folder)
        
        res_model_folder = "../results/train-graphs"
        if not os.path.exists(res_model_folder):
            os.makedirs(res_model_folder)
            
        plt.plot(epochs, train_loss[0:cpt], label='Train', color='blue', marker='x')
        plt.plot(epochs, test_loss[0:cpt], label='Test', color='red', marker='x')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'Training and test loss per epoch for ')
        plt.legend()
        path_name = res_model_folder + f"/DUnet_{loss_type}_Data_lossPerEpoch{epochs}.png"
        plt.savefig(path_name)
        plt.close()
        print(f"Model training and test loss per epoch saved at: \'{path_name}\'")
        
        plt.plot(epochs, train_accuracy[0:cpt], label='Train', color='blue', marker='x')
        plt.plot(epochs, test_accuracy[0:cpt], label='Test', color='red', marker='x')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title(f'Training and test accuracy per epoch for ')
        plt.legend()
        path_name = res_model_folder + f"/DUnet_{loss_type}_Data_accuracyPerEpoch{epochs}.png"
        plt.savefig(path_name)
        plt.close()
        print(f"Model training and test accuracy per epoch saved at: \'{path_name}\'")
        
    # Train the model
    for epoch in range(epochs):
        model.train()

        with tqdm(train_loader) as tepoch:
            for og_imgs, noisy_imgs in tepoch:
                og_imgs, noisy_imgs = og_imgs.to(device), noisy_imgs.to(device)
                optimizer.zero_grad()
                outputs = model(noisy_imgs)

                # Save the last image for debug
                if epoch == epochs-1 and not has_saved_images:
                    has_saved_images = True

                    fig, axes = plt.subplots(1, 3, figsize=(10, 5))
                    axes[0].imshow(noisy_imgs[0].cpu().detach().permute(1, 2, 0))
                    axes[0].set_title("Noisy Image")
                    axes[0].axis('off')
                    axes[1].imshow(outputs[0].cpu().detach().permute(1, 2, 0))
                    axes[1].set_title("Denoised Image")
                    axes[1].axis('off')
                    axes[2].imshow(og_imgs[0].cpu().detach().permute(1, 2, 0))
                    axes[2].set_title("Original Image")
                    axes[2].axis('off')
                    plt.tight_layout()
                    plt.show()
                    
                loss = criterion(outputs, og_imgs)
                loss.backward()
                optimizer.step()
    
    # Eval the model
        if epoch % step_save == 0:
            model.eval()
            with torch.no_grad():
                cpt2 = 0
                total = 0
                for og_imgs, noisy_imgs in train_loader:
                    og_imgs, noisy_imgs = og_imgs.to(device), noisy_imgs.to(device)

                    outputs = model(noisy_imgs)
                    loss = criterion(outputs, og_imgs)

                    train_accuracy_saved[cpt] += (outputs == og_imgs).sum().item()
                    train_loss_saved[cpt] += loss.item()

                    cpt2 += 1
                    total += noisy_imgs.size(0)

                train_loss_saved[cpt] /= cpt2
                train_accuracy_saved[cpt] /= (total*3*width*height)

                cpt2 = 0
                total = 0
                has_saved_images = False
                for og_imgs, noisy_imgs in test_loader:
                    og_imgs, noisy_imgs = og_imgs.to(device), noisy_imgs.to(device)

                    outputs = model(noisy_imgs)
                    loss = criterion(outputs, og_imgs)

                    test_accuracy_saved[cpt] += (outputs == og_imgs).sum().item()
                    test_loss_saved[cpt] += loss.item()

                    cpt2 += 1
                    total += noisy_imgs.size(0)

                    if epoch == epochs-1 and not has_saved_images:
                        has_saved_images = True

                        fig, axes = plt.subplots(1, 3, figsize=(10, 5))
                        axes[0].imshow(noisy_imgs[0].cpu().detach().permute(1, 2, 0))
                        axes[0].set_title("Noisy Image")
                        axes[0].axis('off')
                        axes[1].imshow(outputs[0].cpu().detach().permute(1, 2, 0))
                        axes[1].set_title("Denoised Image")
                        axes[1].axis('off')
                        axes[2].imshow(og_imgs[0].cpu().detach().permute(1, 2, 0))
                        axes[2].set_title("Original Image")
                        axes[2].axis('off')
                        plt.tight_layout()
                        plt.show()

                test_loss_saved[cpt] /= cpt2
                test_accuracy_saved[cpt] /= (total*3*width*height)

                cpt2 = 0

            cpt += 1
        print(f"Epoch {epoch+1}/{epochs}, Loss on train: {train_loss_saved[cpt-1]}, Loss on test: {test_loss_saved[cpt-1]}")
    
    # Save the model
    torch.save(model.state_dict(), f'DUnet_model.pth')
    print(f"Model DUnet successfully saved !")

    # Extract the data
    if device.type == "cuda" and torch.is_tensor(train_loss_saved[0]):
        train_loss_saved = [tensor.cpu().item() for tensor in train_loss_saved]
        test_loss_saved = [tensor.cpu().item() for tensor in test_loss_saved]

    # Save the results of the model
    epochs_ = range(1,epochs+1,step_save)
    save_accuracy_loss_results(epochs_, train_loss_saved, test_loss_saved,
                               train_accuracy_saved, test_accuracy_saved, 
                               cpt)
    
    # Final accuracy test
    print("Final accuracy on test:", test_accuracy_saved[cpt-1])
    return

############################################################################
#                                                                          #
#                                   MAIN                                   #
#                                                                          #
############################################################################

if __name__ == "__main__":
    train_model(loss_type='MSE')