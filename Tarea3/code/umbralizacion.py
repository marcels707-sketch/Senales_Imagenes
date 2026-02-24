import cv2
import matplotlib.pyplot as plt
from skimage import filters
import numpy as np

# Ruta de la imagen
ruta_archivo = "C:/Users/Marce/Documents/SenalesImagenes/Laboratorios/Tarea3/data/1734589457679.tif"

#Cargar la imagen en escala de grises
img_gris = cv2.imread(ruta_archivo, cv2.IMREAD_UNCHANGED)

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
    original = recorte.astype("uint8")
    gaussiano = (filters.gaussian(recorte, sigma=2) * 255).astype("uint8")

    #morfologia
    kernel = np.ones((5, 5), np.uint8)

    #original
    apertura_orig = cv2.morphologyEx(original, cv2.MORPH_OPEN, kernel)
    dilat_orig = cv2.dilate(original, kernel, iterations=1)
    tophat_orig = cv2.morphologyEx(original, cv2.MORPH_TOPHAT, kernel)

    #gaussiano
    apertura_gauss = cv2.morphologyEx(gaussiano, cv2.MORPH_OPEN, kernel)
    dilat_gauss = cv2.dilate(gaussiano, kernel, iterations=1)
    tophat_gauss = cv2.morphologyEx(gaussiano, cv2.MORPH_TOPHAT, kernel)

    #umbralizacion otsu
    def otsu_u8(img):
        t = filters.threshold_otsu(img)
        mask = img > t
        # Limpieza post-umbral
        mask = cv2.morphologyEx(mask.astype("uint8")*255, cv2.MORPH_OPEN, kernel)
        return mask

    # Otsu sobre original + morfología
    otsu_orig = otsu_u8(original)
    otsu_apertura_orig = otsu_u8(apertura_orig)
    otsu_dilat_orig = otsu_u8(dilat_orig)
    otsu_tophat_orig = otsu_u8(tophat_orig)

    # Otsu sobre gaussiano + morfología
    otsu_gauss = otsu_u8(gaussiano)
    otsu_apertura_gauss = otsu_u8(apertura_gauss)
    otsu_dilat_gauss = otsu_u8(dilat_gauss)
    otsu_tophat_gauss = otsu_u8(tophat_gauss)

    #umbralizacion adaptativa
    def adapt_smooth(img):
        t = filters.threshold_local(img, block_size=101, offset=3)
        mask = img > t
        mask = cv2.morphologyEx(mask.astype("uint8")*255, cv2.MORPH_OPEN, kernel)
        return mask

    adapt_orig = adapt_smooth(original)
    adapt_apertura_orig = adapt_smooth(apertura_orig)
    adapt_dilat_orig = adapt_smooth(dilat_orig)
    adapt_tophat_orig = adapt_smooth(tophat_orig)

    adapt_gauss = adapt_smooth(gaussiano)
    adapt_apertura_gauss = adapt_smooth(apertura_gauss)
    adapt_dilat_gauss = adapt_smooth(dilat_gauss)
    adapt_tophat_gauss = adapt_smooth(tophat_gauss)

    fig, axes = plt.subplots(4, 4, figsize=(14, 10))
    fig.suptitle("Morfología + Otsu + Adaptativo (Optimizado)", fontsize=16, fontweight='bold')

    fila1 = [
        otsu_orig,
        otsu_apertura_orig,
        otsu_dilat_orig,
        otsu_tophat_orig
    ]
    fila1_titles = [
        "Otsu(Original)",
        "Otsu(Apertura)",
        "Otsu(Dilatación)",
        "Otsu(Top-hat)"
    ]

    for i in range(4):
        axes[0, i].imshow(fila1[i], cmap='gray')
        axes[0, i].set_title(fila1_titles[i])
        axes[0, i].axis('off')

    # FILA 2 
    fila2 = [
        adapt_orig,
        adapt_apertura_orig,
        adapt_dilat_orig,
        adapt_tophat_orig
    ]
    fila2_titles = [
        "Adapt(Original)",
        "Adapt(Apertura)",
        "Adapt(Dilatación)",
        "Adapt(Top-hat)"
    ]

    for i in range(4):
        axes[1, i].imshow(fila2[i], cmap='gray')
        axes[1, i].set_title(fila2_titles[i])
        axes[1, i].axis('off')

    # FILA 3
    fila3 = [
        otsu_gauss,
        otsu_apertura_gauss,
        otsu_dilat_gauss,
        otsu_tophat_gauss
    ]
    fila3_titles = [
        "Otsu(Gauss)",
        "Otsu(Apertura G)",
        "Otsu(Dilatación G)",
        "Otsu(Top-hat G)"
    ]

    for i in range(4):
        axes[2, i].imshow(fila3[i], cmap='gray')
        axes[2, i].set_title(fila3_titles[i])
        axes[2, i].axis('off')

    # FILA 4
    fila4 = [
        adapt_gauss,
        adapt_apertura_gauss,
        adapt_dilat_gauss,
        adapt_tophat_gauss
    ]
    fila4_titles = [
        "Adapt(Gauss)",
        "Adapt(Apertura G)",
        "Adapt(Dilatación G)",
        "Adapt(Top-hat G)"
    ]

    for i in range(4):
        axes[3, i].imshow(fila4[i], cmap='gray')
        axes[3, i].set_title(fila4_titles[i])
        axes[3, i].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()