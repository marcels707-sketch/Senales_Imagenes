import cv2
import matplotlib.pyplot as plt
from skimage import filters, segmentation, measure
from scipy import ndimage as ndi
import numpy as np

# Ruta de la imagen
ruta_archivo = "C:/Users/Marce/Documents/SenalesImagenes/Laboratorios/Tarea3/data/1734589457679.tif"

# Cargar la imagen en escala de grises
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

    #base
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
        mask = cv2.morphologyEx(mask.astype("uint8")*255, cv2.MORPH_OPEN, kernel)
        return mask

    # Otsu original
    otsu_list_orig = [
        otsu_u8(original),
        otsu_u8(apertura_orig),
        otsu_u8(dilat_orig),
        otsu_u8(tophat_orig)
    ]

    # Otsu gauss
    otsu_list_gauss = [
        otsu_u8(gaussiano),
        otsu_u8(apertura_gauss),
        otsu_u8(dilat_gauss),
        otsu_u8(tophat_gauss)
    ]

    # umbralizacion adaptativo
    def adapt_smooth(img):
        t = filters.threshold_local(img, block_size=101, offset=3)
        mask = img > t
        mask = cv2.morphologyEx(mask.astype("uint8")*255, cv2.MORPH_OPEN, kernel)
        return mask

    adapt_list_orig = [
        adapt_smooth(original),
        adapt_smooth(apertura_orig),
        adapt_smooth(dilat_orig),
        adapt_smooth(tophat_orig)
    ]

    adapt_list_gauss = [
        adapt_smooth(gaussiano),
        adapt_smooth(apertura_gauss),
        adapt_smooth(dilat_gauss),
        adapt_smooth(tophat_gauss)
    ]

    #watershed
    def watershed_seg(img):
        img_uint = img.astype("uint8")

        t = filters.threshold_otsu(img_uint)
        binaria = img_uint > t

        distancia = ndi.distance_transform_edt(binaria)
        marcadores = measure.label(distancia > 0.4 * distancia.max())
        gradiente = filters.sobel(img_uint)

        ws = segmentation.watershed(gradiente, marcadores, mask=binaria)
        return ws

    ws_list_orig = [
        watershed_seg(original),
        watershed_seg(apertura_orig),
        watershed_seg(dilat_orig),
        watershed_seg(tophat_orig)
    ]

    ws_list_gauss = [
        watershed_seg(gaussiano),
        watershed_seg(apertura_gauss),
        watershed_seg(dilat_gauss),
        watershed_seg(tophat_gauss)
    ]


    fig, axes = plt.subplots(4, 8, figsize=(28, 12))
    fig.suptitle("Otsu + Adaptativo + Watershed (Original y Gaussiano)", fontsize=18, fontweight='bold')

    col_titles = [
        "Orig", "Apertura", "Dilatación", "Top-hat",
        "Adapt Orig", "Adapt Apertura", "Adapt Dilatación", "Adapt Top-hat"
    ]

    # FILA 1
    fila1 = otsu_list_orig + adapt_list_orig
    for i in range(8):
        axes[0, i].imshow(fila1[i], cmap='gray')
        axes[0, i].set_title(col_titles[i])
        axes[0, i].axis('off')

    # FILA 2
    fila2 = otsu_list_gauss + adapt_list_gauss
    for i in range(8):
        axes[1, i].imshow(fila2[i], cmap='gray')
        axes[1, i].set_title(col_titles[i])
        axes[1, i].axis('off')

    # FILA 3
    fila3 = ws_list_orig + adapt_list_orig
    for i in range(8):
        axes[2, i].imshow(fila3[i], cmap='nipy_spectral')
        axes[2, i].set_title(col_titles[i])
        axes[2, i].axis('off')

    # FILA 4
    fila4 = ws_list_gauss + adapt_list_gauss
    for i in range(8):
        axes[3, i].imshow(fila4[i], cmap='nipy_spectral')
        axes[3, i].set_title(col_titles[i])
        axes[3, i].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()