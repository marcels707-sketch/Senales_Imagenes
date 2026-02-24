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

    # Obtener lista de imágenes
    image_files = [f for f in os.listdir(folder_path)
                   if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif"))]

    total = len(image_files)
    cols = 3
    rows = int(np.ceil(total / cols))

    # Aumentamos tamaño de figura para que las imágenes sean más grandes
    plt.figure(figsize=(18, 6 * rows))

    for idx, file_name in enumerate(image_files):

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

        # Si hay cluster objetivo, resaltarlo
        if target_cluster is not None and target_cluster < len(centers):
            highlight = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            highlight[labels == target_cluster] = [0, 255, 0]
            result = highlight
        else:
            result = cv2.cvtColor(colored_img, cv2.COLOR_BGR2RGB)

        # Mostrar en el panel
        ax = plt.subplot(rows, cols, idx + 1)
        ax.imshow(result)
        ax.set_title(file_name, fontsize=8, loc='left')  # texto pequeño y alineado a la izquierda
        ax.axis('off')

    plt.tight_layout()
    plt.show()


# ============================================================
# EJECUCIÓN
# ============================================================

test_folder = "C:/Users/Marce/Documents/SenalesImagenes/Clase3/proyecto/dataset/test"

# Tus centroides reales
centroides_entrenados = [81.00, 83.33, 86.00, 88.82, 91.42]

apply_fixed_clusters_to_folder(
    test_folder,
    centers=centroides_entrenados,
    target_cluster=3
)