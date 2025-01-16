# Projet Am4ip - Corentin Seutin & Iban Oyharcabal

Ce projet implémente des modèles de segmentation d'images pour diverses applications, utilisant des architectures comme Unet et PSPNet. Le projet utilise des outils comme PyTorch et des bibliothèques associées pour l'entraînement, l'évaluation, et la visualisation des résultats.

## Table des matières
1. [Prérequis](#prérequis)
2. [Installation](#installation)
3. [Utilisation](#utilisation)
4. [Contribuer](#contribuer)
5. [Licences](#licences)
6. [Contact](#contact)

## Prérequis

Avant de commencer, assurez-vous d'avoir installé les bibliothèques suivantes :

- PyTorch
- NumPy
- OpenCV
- Matplotlib
- PIL (Pillow)
- tqdm

## Installation

1. Créez un environnement virtuel :

    ```bash
    python -m venv env
    source env/bin/activate   # Sur Windows, utilisez env\Scripts\activate
    ```

2. Installez les dépendances depuis le fichier `requirements.txt` :

    ```bash
    pip install -r requirements.txt
    ```

## Utilisation

1. Récupérez les différents datasets à l'aide du lien suivant :

https://drive.google.com/drive/folders/13ovXtnx_hT4vpPW4EeHAjQLJEa3FeQpm

2. Décomposition de l'archive et modification des chemins d'accès aux datasets :

- *Night-debruitees* : images rainy débruitées à l'aide du modèle pré-entraîné PSPNet. Modifiez la ligne 41 *denoised_image_path =* du fichier *./src/am4ip/dataset.py* pour y mettre le chemin d'accès au dossier *Night-debruitees/rainy_images/*. 

- *project-dataset* : contient le reste des datasets et notamment les datasets : *cache et cache_night* qui stocke les images prétraitées avant l'entraînement d'un modèle. *night* correspond au dataset proposé par Wu et al. comme mentionné dans le rapport, il correspond au dataset *Night-Rainy* proposé. *night-noisy* correspond au dataset bruité à l'aide du script *construct_fake_dataset.py* utilisé pour l'entraînement du modèle U-Net pour le débruitage. Le reste des sous-dossiers correspond aux datasets fournis pour le projet. Modifiez les lignes 39, 41 et 136 (si vous êtes au CREMI, sinon modifiez les lignes 44, 45 et 136) du fichier *./src/am4ip/dataset.py* pour y mettre les chemins d'accès aux dossiers *project-dataset* et *cache*. 

3. Récupérez le dépôt modifié de l'architecture IBCLN proposée par Li et al. à l'aide du lien suivant : 

https://github.com/CorentinSeutin/Light-halo-denoiser

Ce dépôt permet de débruiter les images *rainy* du dataset du projet et stocke les résultats 
au chemin suivant : *Light-halo-denoiser\datasets\wo-reflection\rainy_images*.

3. Exécutez les scripts pour l'entraînement ou l'évaluation des modèles. Par exemple :

    ```bash
    python construct_fake_dataset.py
    ```

## Description des différents fichiers python

### Fichiers du projet

- *./script/construct_fake_dataset.py* : script permettant de construire un faux dataset pour l'entraînement d'un modèle U-Net utilisé pour le débruitage des halos lumineux du dataset *rainy*.

- *./script/test-denoiser.py* : script permettant de tester le modèle U-Net entraîné pour le débruitage. Utilise le modèle pré-entraîné *./script.DUnet_model.pth*.

- *./script/test.py* : script permettant de tester les modèles entraînés pour la segmentation des datasets *sunny* et *rainy* contenus dans le dossier *./script/models/*.

- *./script/tools.py* : fichier contenant des fonctions utiles pour le pré et post-traitement des données.

- *./script/train_denoiser.py* : script permettant d'entraîner un modèle U-Net à débruiter des images bruitées (halos lumineux) créées au préalable à l'aide du script *construct_fake_dataset.py*.

- *./script/train.py* : script permettant d'entraîner les modèles U-Net et PSPNet sur différentes fonctions de perte (Dice Loss et Cross-entropy) et de stocker les courbes associées à l'entraînement (loss, accuracy et MIoU moyennes en fonction du nombre d'epochs).

- *./src/am4ip/dataset.py* : fichier contenant la classe *ProjectDataset* et *NightDataset* utiles pour "dataloader" les différents datasets.

- *./src/am4ip/losses.py* : fichier contenant les différentes fonctions de perte utilisées (Dice Loss).

- *./src/am4ip/metrics.py* : fichier contenant les différentes métriques utilisées (notamment la MIoU).

- *./src/am4ip/models.py* : fichier contenant les différentes architectures de modèles utilisés.

### Fichiers de dépôt *Light-halo-denoiser*

Pour tester le débruitage : 

1. Créer un dossier : *./datasets/reflection/*.

2. Créer les sous-dossiers : *./datasets/reflection/testA1/*, *./datasets/reflection/testA2/* et *./datasets/reflection/testB/*. 

3. Copier/coller l'ensemble des images du dataset *rainy* dans le sous-dossier *./datasets/reflection/testB/*.

4. Exécuter le script : *./process_rainy.py*.

5. Retrouver les résultats dans le dossier : *./datasets/wo-reflection/*.