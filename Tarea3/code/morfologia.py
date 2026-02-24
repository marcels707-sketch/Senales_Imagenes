import cv2
import matplotlib.pyplot as plt
from skimage import filters
import numpy as np

# Ruta de la imagen
ruta_archivo = "C:/Users/Marce/Documents/SenalesImagenes/Laboratorios/Tarea3/data/1734589457679.tif"

#Cargar la imagen en escala de grises
img_gris = cv2.imread(ruta_archivo, cv2.IMREAD_GRAYSCALE)

if img_gris is None:
    print(f"Error: No se encontró la imagen en {ruta_archivo}")
else:
    alto, ancho = img_gris.shape
    print(f"Dimensiones: {ancho}px de ancho x {alto}px de alto")

    # Coordenadas del recorte
    x1, y1 = 0, 0
    x2, y2 = 640, 350

    # Seguridad para no salir de los límites
    x1 = max(0, min(x1, ancho - 1))
    x2 = max(0, min(x2, ancho))
    y1 = max(0, min(y1, alto - 1))
    y2 = max(0, min(y2, alto))

    recorte = img_gris[y1:y2, x1:x2]

    #filtros base
    original = recorte
    gaussiano = filters.gaussian(recorte, sigma=1)

    # Convertir gaussiano a uint8 para OpenCV
    gauss_uint8 = (gaussiano * 255).astype("uint8")

    #morfologia
    kernel = np.ones((5, 5), np.uint8)

    # Original
    apertura_orig = cv2.morphologyEx(original, cv2.MORPH_OPEN, kernel)
    dilat_orig = cv2.dilate(original, kernel, iterations=1)
    tophat_orig = cv2.morphologyEx(original, cv2.MORPH_TOPHAT, kernel)

    # Gaussiano
    apertura_gauss = cv2.morphologyEx(gauss_uint8, cv2.MORPH_OPEN, kernel)
    dilat_gauss = cv2.dilate(gauss_uint8, kernel, iterations=1)
    tophat_gauss = cv2.morphologyEx(gauss_uint8, cv2.MORPH_TOPHAT, kernel)

    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    fig.suptitle("Morfología con OpenCV (Apertura, Dilatación, Top-hat)", fontsize=16, fontweight='bold')

    # FILA 1
    fila1 = [
        original,
        apertura_orig,
        dilat_orig,
        tophat_orig
    ]

    fila1_titles = [
        "Original",
        "Apertura (OpenCV)",
        "Dilatación (OpenCV)",
        "Top-hat (OpenCV)"
    ]

    for i in range(4):
        axes[0, i].imshow(fila1[i], cmap='gray')
        axes[0, i].set_title(fila1_titles[i])
        axes[0, i].axis('off')

    # FILA 2
    fila2 = [
        gauss_uint8,
        apertura_gauss,
        dilat_gauss,
        tophat_gauss
    ]

    fila2_titles = [
        "Gaussiano",
        "Apertura G (OpenCV)",
        "Dilatación G (OpenCV)",
        "Top-hat G (OpenCV)"
    ]

    for i in range(4):
        axes[1, i].imshow(fila2[i], cmap='gray')
        axes[1, i].set_title(fila2_titles[i])
        axes[1, i].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()