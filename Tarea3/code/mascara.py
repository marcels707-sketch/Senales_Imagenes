import cv2
import matplotlib.pyplot as plt
from skimage import filters, segmentation, measure
from scipy import ndimage as ndi
import numpy as np

# Ruta de la imagen
ruta_archivo = "C:/Users/Marce/Documents/SenalesImagenes/Laboratorios/Tarea3/data/1734589457679.tif"

# Cargar la imagen
img_gris = cv2.imread(ruta_archivo, cv2.IMREAD_UNCHANGED)

if img_gris is None:
    print(f"Error: No se encontró la imagen en {ruta_archivo}")
else:
    alto, ancho = img_gris.shape
    print(f"Dimensiones: {ancho}px x {alto}px")

    # Recorte
    recorte = img_gris[0:350, 0:640]

    # ============================================================
    # 1. GENERAR MÁSCARA (SIN ENTRENAMIENTO)
    # ============================================================
    # Usamos Otsu sobre el recorte para generar la máscara
    t_mask = filters.threshold_otsu(recorte)
    mask = (recorte > t_mask).astype("uint8") * 255

    # Suavizamos la máscara
    kernel_mask = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_mask)

    # Función para aplicar máscara
    def aplicar_mascara(img):
        return cv2.bitwise_and(img, img, mask=mask)

    # ============================================================
    # 2. FILTROS BASE
    # ============================================================
    original = recorte.astype("uint8")
    gaussiano = (filters.gaussian(recorte, sigma=2) * 255).astype("uint8")

    # Aplicar máscara
    original_m = aplicar_mascara(original)
    gauss_m = aplicar_mascara(gaussiano)

    # ============================================================
    # 3. MORFOLOGÍA
    # ============================================================
    kernel = np.ones((5, 5), np.uint8)

    # Original
    apertura_orig = aplicar_mascara(cv2.morphologyEx(original, cv2.MORPH_OPEN, kernel))
    dilat_orig = aplicar_mascara(cv2.dilate(original, kernel, iterations=1))
    tophat_orig = aplicar_mascara(cv2.morphologyEx(original, cv2.MORPH_TOPHAT, kernel))

    # Gaussiano
    apertura_gauss = aplicar_mascara(cv2.morphologyEx(gaussiano, cv2.MORPH_OPEN, kernel))
    dilat_gauss = aplicar_mascara(cv2.dilate(gaussiano, kernel, iterations=1))
    tophat_gauss = aplicar_mascara(cv2.morphologyEx(gaussiano, cv2.MORPH_TOPHAT, kernel))

    # ============================================================
    # 4. OTSU
    # ============================================================
    def otsu_u8(img):
        t = filters.threshold_otsu(img)
        mask_bin = img > t
        return cv2.morphologyEx(mask_bin.astype("uint8")*255, cv2.MORPH_OPEN, kernel)

    otsu_list_orig = [
        otsu_u8(original_m),
        otsu_u8(apertura_orig),
        otsu_u8(dilat_orig),
        otsu_u8(tophat_orig)
    ]

    otsu_list_gauss = [
        otsu_u8(gauss_m),
        otsu_u8(apertura_gauss),
        otsu_u8(dilat_gauss),
        otsu_u8(tophat_gauss)
    ]

    # ============================================================
    # 5. ADAPTATIVO
    # ============================================================
    def adapt_smooth(img):
        t = filters.threshold_local(img, block_size=101, offset=3)
        mask_bin = img > t
        return cv2.morphologyEx(mask_bin.astype("uint8")*255, cv2.MORPH_OPEN, kernel)

    adapt_list_orig = [
        adapt_smooth(original_m),
        adapt_smooth(apertura_orig),
        adapt_smooth(dilat_orig),
        adapt_smooth(tophat_orig)
    ]

    adapt_list_gauss = [
        adapt_smooth(gauss_m),
        adapt_smooth(apertura_gauss),
        adapt_smooth(dilat_gauss),
        adapt_smooth(tophat_gauss)
    ]

    # ============================================================
    # 6. WATERSHED
    # ============================================================
    def watershed_seg(img):
        img_uint = img.astype("uint8")
        t = filters.threshold_otsu(img_uint)
        binaria = img_uint > t
        distancia = ndi.distance_transform_edt(binaria)
        marcadores = measure.label(distancia > 0.4 * distancia.max())
        gradiente = filters.sobel(img_uint)
        return segmentation.watershed(gradiente, marcadores, mask=binaria)

    ws_list_orig = [
        watershed_seg(original_m),
        watershed_seg(apertura_orig),
        watershed_seg(dilat_orig),
        watershed_seg(tophat_orig)
    ]

    ws_list_gauss = [
        watershed_seg(gauss_m),
        watershed_seg(apertura_gauss),
        watershed_seg(dilat_gauss),
        watershed_seg(tophat_gauss)
    ]

    # ============================================================
    # 7. PANEL 4×8
    # ============================================================
    fig, axes = plt.subplots(4, 8, figsize=(28, 12))
    fig.suptitle("Máscara aplicada + Otsu + Adaptativo + Watershed", fontsize=18, fontweight='bold')

    col_titles = [
        "Orig", "Apertura", "Dilatación", "Top-hat",
        "Adapt Orig", "Adapt Apertura", "Adapt Dilatación", "Adapt Top-hat"
    ]

    # FILA 1 — OTSU ORIGINAL + ADAPT ORIGINAL
    fila1 = otsu_list_orig + adapt_list_orig
    for i in range(8):
        axes[0, i].imshow(fila1[i], cmap='gray')
        axes[0, i].set_title(col_titles[i])
        axes[0, i].axis('off')

    # FILA 2 — OTSU GAUSS + ADAPT GAUSS
    fila2 = otsu_list_gauss + adapt_list_gauss
    for i in range(8):
        axes[1, i].imshow(fila2[i], cmap='gray')
        axes[1, i].set_title(col_titles[i])
        axes[1, i].axis('off')

    # FILA 3 — WS ORIGINAL
    fila3 = ws_list_orig + adapt_list_orig
    for i in range(8):
        axes[2, i].imshow(fila3[i], cmap='nipy_spectral')
        axes[2, i].set_title(col_titles[i])
        axes[2, i].axis('off')

    # FILA 4 — WS GAUSS
    fila4 = ws_list_gauss + adapt_list_gauss
    for i in range(8):
        axes[3, i].imshow(fila4[i], cmap='nipy_spectral')
        axes[3, i].set_title(col_titles[i])
        axes[3, i].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.show()

    