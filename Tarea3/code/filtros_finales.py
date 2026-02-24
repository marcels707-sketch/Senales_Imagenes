import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage import filters, segmentation, measure
from scipy import ndimage as ndi

def apply_fixed_clusters_to_folder(folder_path, centers, target_cluster=None):

    # Ordenar centroides
    centers = np.array(centers, dtype=np.float32)
    centers = centers[np.argsort(centers)]

    image_files = [f for f in os.listdir(folder_path)
                   if f.lower().endswith(".tif")][:5]

    rows = len(image_files)
    cols = 6  # Original + Clusters + Gauss + Dilat + Adapt + Watershed

    plt.figure(figsize=(26, 5 * rows))

    for row, file_name in enumerate(image_files):

        # Cargar 
        img_path = os.path.join(folder_path, file_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Asignar clusters
        pixel_values = img.astype(np.float32)
        distances = np.abs(pixel_values[:, :, np.newaxis] - centers)
        labels = np.argmin(distances, axis=2)

        # Imagen coloreada
        h, w = img.shape
        colored_img = np.zeros((h, w, 3), dtype=np.uint8)

        colors = [
            [255, 0, 0], [0, 255, 0], [0, 0, 255],
            [255, 255, 0], [255, 0, 255]
        ]

        for i in range(len(centers)):
            colored_img[labels == i] = colors[i]


        # Máscara del cluster objetivo
        mask = (labels == target_cluster).astype("uint8") * 255

        def aplicar_mascara(im):
            return cv2.bitwise_and(im, im, mask=mask)

        # Filtro Gaussiano
        gauss = (filters.gaussian(img, sigma=2) * 255).astype("uint8")
        gauss_m = aplicar_mascara(gauss)

        # Dilatación

        kernel = np.ones((5, 5), np.uint8)
        dilat = aplicar_mascara(cv2.dilate(img, kernel, iterations=1))

        # Umbralización adaptativa
        def adapt_smooth(im):
            t = filters.threshold_local(im, block_size=101, offset=3)
            m = im > t
            return m.astype("uint8") * 255

        adapt = adapt_smooth(aplicar_mascara(img))

        # Watershed

        def watershed_seg(im):
            im_uint = im.astype("uint8")
            t = filters.threshold_otsu(im_uint)
            binaria = im_uint > t
            distancia = ndi.distance_transform_edt(binaria)
            marcadores = measure.label(distancia > 0.4 * distancia.max())
            grad = filters.sobel(im_uint)
            return segmentation.watershed(grad, marcadores, mask=binaria)

        ws = watershed_seg(aplicar_mascara(img))


        panel = [
            img,
            cv2.cvtColor(colored_img, cv2.COLOR_BGR2RGB),
            gauss_m,
            dilat,
            adapt,
            ws
        ]

        titles = [
            "Original",
            "Clusters",
            "Gaussiano",
            "Dilatación",
            "Adaptativo",
            "Watershed"
        ]

        for col, (img_out, title) in enumerate(zip(panel, titles)):
            ax = plt.subplot(rows, cols, row * cols + col + 1)
            ax.imshow(img_out, cmap='gray' if col != 1 else None)
            ax.set_title(f"{file_name}\n{title}", fontsize=8, loc='left')
            ax.axis('off')

    plt.tight_layout()
    plt.show()


test_folder = "C:/Users/Marce/Documents/SenalesImagenes/Clase3/proyecto/dataset/test"
centroides_entrenados = [81.00, 83.33, 86.00, 88.82, 91.42]

apply_fixed_clusters_to_folder(
    test_folder,
    centers=centroides_entrenados,
    target_cluster=3
)