import cv2
import numpy as np
import random
import os
from tqdm import tqdm  # Import de tqdm

def keep_some_imgs(input_dir):
    # Lister tous les fichiers .jpg dans le répertoire
    images = [f for f in os.listdir(input_dir) if f.endswith('.jpg')]
    
    # Vérifier si il y a plus de 500 images
    if len(images) > 500:
        # Mélanger les fichiers pour supprimer les images de manière aléatoire
        random.shuffle(images)
        
        # Supprimer les images sauf les 500 premières
        for img_name in images[500:]:
            img_path = os.path.join(input_dir, img_name)
            os.remove(img_path)
            print(f"Image supprimée: {img_name}")

def add_halo(image, position, radius, intensity, color=None):
    overlay = image.copy()
    output = image.copy()
    h, w, _ = image.shape

    # Calculer white_ratio en fonction de radius (plus radius est petit, plus white_ratio est grand)
    white_ratio = min(1, max(0.5, 1 - radius / 50))  # Ajustez les constantes pour affiner le comportement
    color_ratio = 1 - white_ratio
    second_radius = radius * white_ratio

    # Si aucune couleur n'est spécifiée, choisir une couleur aléatoire (par exemple, orange/jaune)
    if color is None:
        color = np.array([random.randint(125, 255), random.randint(125, 255), random.randint(125, 255)])  # Orange/Jaune

    for i in range(h):
        for j in range(w):
            # Calculer la distance euclidienne du pixel au centre du halo
            distance = np.sqrt((i - position[1])**2 + (j - position[0])**2)
            
            if distance < radius:
                # Ajouter l'intensité en mélangeant avec la couleur du halo
                if distance / radius < white_ratio:
                    factor = (distance / second_radius)  # Facteur de dégradé entre 0 et 1
                    white_value = int(255 - (25 * factor))  # Transition de 255 à 240
                    overlay[i, j] = np.array([
                        white_value,
                        white_value,
                        white_value
                    ])
                else:
                    # Calculer l'intensité en fonction de la distance (plus la distance est grande, plus l'intensité est faible)
                    halo_intensity = intensity * (1 - (distance - second_radius) / (radius - second_radius))
                    overlay[i, j] = np.array([
                        min(255, image[i, j][0] + int(color[0] * halo_intensity / 255)),  # Canal R
                        min(255, image[i, j][1] + int(color[1] * halo_intensity / 255)),  # Canal G
                        min(255, image[i, j][2] + int(color[2] * halo_intensity / 255))   # Canal B
                    ])

    # Appliquer un effet de mélange (transparence)
    alpha = 1  # Ajustez l'opacité ici
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    return output

def generate_dataset(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    images = [f for f in os.listdir(input_dir) if f.endswith('.jpg')]
    
    for img_name in tqdm(images, desc="Génération des images bruitées", unit="image"):
        img_path = os.path.join(input_dir, img_name)
        image = cv2.imread(img_path)
        h, w, _ = image.shape
        
        for _ in range(random.randint(1, 5)):
            x = random.randint(0, w)
            y = random.randint(0, h)
            radius = random.randint(5, 50)
            intensity = random.randint(100, 255)
            image = add_halo(image, (x, y), radius, intensity)
        
        output_path = os.path.join(output_dir, img_name)
        cv2.imwrite(output_path, image)

#keep_some_imgs(input_dir="C:\\Users\\coren\\Desktop\\project-dataset\\night\\")

# Appel de la fonction
generate_dataset(input_dir="C:\\Users\\coren\\Desktop\\project-dataset\\night\\", 
                 output_dir="C:\\Users\\coren\\Desktop\\project-dataset\\night-noisy\\")
