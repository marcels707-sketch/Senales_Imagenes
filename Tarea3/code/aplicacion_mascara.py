import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def apply_fixed_clusters_to_folder(folder_path, centers, target_cluster=None):

    # Ordenar centroides
    centers = np.array(centers, dtype=np.float32)
    sorted_idx = np.argsort(centers)
    centers = centers[sorted_idx]

    # Colores para visualizar clusters
    colors = [
        [255, 0, 0],    # Azul
        [0, 255, 0],    # Verde
        [0, 0, 255],    # Rojo
        [255, 255, 0],  # Cyan
        [255, 0, 255]   # Magenta
    ]

    # Tomar solo 5 imágenes
    image_files = [f for f in os.listdir(folder_path)
                   if f.lower().endswith((".tif"))][:5]

    rows = len(image_files)
    cols = 3

    # Figura grande para buena visualización
    plt.figure(figsize=(18, 5 * rows))

    for row, file_name in enumerate(image_files):

        img_path = os.path.join(folder_path, file_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            print(f"No se pudo cargar {file_name}")
            continue

        # Recorte igual que tu pipeline
        img = img[0:350, 0:640]

        # Asignar cada pixel al centroide más cercano
        pixel_values = img.astype(np.float32)
        distances = np.abs(pixel_values[:, :, np.newaxis] - centers)
        labels = np.argmin(distances, axis=2)

        # Imagen coloreada
        h, w = img.shape
        colored_img = np.zeros((h, w, 3), dtype=np.uint8)

        for i in range(len(centers)):
            colored_img[labels == i] = colors[i]

        # Resaltar cluster objetivo
        if target_cluster is not None and target_cluster < len(centers):
            highlight = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            highlight[labels == target_cluster] = [0, 255, 0]
        else:
            highlight = cv2.cvtColor(colored_img, cv2.COLOR_BGR2RGB)

        # --- Columna 1: Imagen original ---
        ax1 = plt.subplot(rows, cols, row * cols + 1)
        ax1.imshow(img, cmap='gray')
        ax1.set_title(file_name, fontsize=8, loc='left')
        ax1.axis('off')

        # --- Columna 2: Segmentación por clusters ---
        ax2 = plt.subplot(rows, cols, row * cols + 2)
        ax2.imshow(cv2.cvtColor(colored_img, cv2.COLOR_BGR2RGB))
        ax2.set_title("Clusters", fontsize=8, loc='left')
        ax2.axis('off')

        # --- Columna 3: Cluster resaltado ---
        ax3 = plt.subplot(rows, cols, row * cols + 3)
        ax3.imshow(highlight)
        ax3.set_title(f"Cluster {target_cluster}", fontsize=8, loc='left')
        ax3.axis('off')

    plt.tight_layout()
    plt.show()


test_folder = "C:/Users/Marce/Documents/SenalesImagenes/Clase3/proyecto/dataset/test"

# Tus centroides reales
centroides_entrenados = [81.00, 83.33, 86.00, 88.82, 91.42]

apply_fixed_clusters_to_folder(
    test_folder,
    centers=centroides_entrenados,
    target_cluster=3
)