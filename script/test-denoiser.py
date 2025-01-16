#source /net/ens/DeepLearning/python3/tensorflow2/bin/activate
#python -m ipykernel install --user --name=tensorflow2
#export PYTHONPATH=$PYTHONPATH:/autofs/unitytravail/travail/coseutin/m2-iis-coco-et-ibaba/Am4ip/projet/src

import torch
import matplotlib.pyplot as plt
import sys

#sys.path.append("/autofs/unitytravail/travail/coseutin/m2-iis-coco-et-ibaba/Am4ip/projet/src")
sys.path.append("/autofs/unitytravail/travail/ioyharcabal/M2S1/m2-iis-coco-et-ibaba/Am4ip/projet/src")
#sys.path.append(os.path.abspath("C:/Users/coren/Desktop/m2-iis-coco-et-ibaba/Am4ip/projet/src"))

from torch.utils.data import DataLoader
from torchvision import transforms

from am4ip.dataset import ProjectDataset
from am4ip.models import DenoiserUnet

from typing import Callable, Optional, Tuple, Union, Literal

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print(device)

############################################################################
#                                                                          #
#                                  TRAIN                                   #
#                                                                          #
############################################################################

def test_model(
                dataset_type : Literal["sunny", "rainy"] = "rainy",
                width : int = 256, height : int = 256,
                data_fraction : float = 0.5
                ) -> None :
    model = DenoiserUnet()
    model_path = "DUnet_model.pth"
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.to(device)
    
    # Dataset 
    dataset = ProjectDataset(
        dataset_type = dataset_type,
        transform=transforms.Compose([
            transforms.Resize((width, height)),
            transforms.ToTensor()
        ]),
        preprocess=True,
        fraction=data_fraction
    )
    
    # Load Datasets
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
        
    # Eval the model
    with torch.no_grad():
        for og_imgs, _ in loader:
            og_imgs = og_imgs.to(device)
            outputs = model(og_imgs)

            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(og_imgs[0].cpu().detach().permute(1, 2, 0))
            axes[0].set_title("Noisy Image")
            axes[0].axis('off')
            axes[1].imshow(outputs[0].cpu().detach().permute(1, 2, 0))
            axes[1].set_title("Denoised Image")
            axes[1].axis('off')
            plt.tight_layout()
            plt.show()

############################################################################
#                                                                          #
#                                   MAIN                                   #
#                                                                          #
############################################################################

if __name__ == "__main__":
    test_model(dataset_type='rainy', data_fraction=0.025)