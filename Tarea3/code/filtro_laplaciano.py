import cv2
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import feature, filters, morphology
from PIL import Image

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

    #Original y el recorte
    fila0_imgs = [img_gris, recorte]
    fila0_titles = ["Imagen Completa", "Recorte (0:350, 0:640)"]

    #Filtrado base
    original = recorte
    laplaciano = ndi.laplace(recorte)
    gaussiano = filters.gaussian(recorte, sigma=1)
    log = ndi.laplace(gaussiano)

    fila1_imgs = [original, laplaciano, gaussiano, log]
    fila1_titles = ["Original", "Laplaciano", "Gaussiano", "LoG"]

    #Sobel
    sobel_imgs = [filters.sobel(img) for img in fila1_imgs]
    sobel_titles = [f"Sobel({t})" for t in fila1_titles]

    #Prewitt
    prewitt_imgs = [filters.prewitt(img) for img in fila1_imgs]
    prewitt_titles = [f"Prewitt({t})" for t in fila1_titles]

    #Canny
    canny_imgs = [feature.canny(img, sigma=1) for img in fila1_imgs]
    canny_titles = [f"Canny({t})" for t in fila1_titles]

   
    fig, axes = plt.subplots(5, 4, figsize=(6, 12))
    fig.suptitle("Filtros, Detección de Bordes y Morfología Matemática", fontsize=18, fontweight='bold')

    axes[0, 0].imshow(fila0_imgs[0], cmap='gray')
    axes[0, 0].set_title(fila0_titles[0])
    axes[0, 0].axis('off')

    axes[0, 1].imshow(fila0_imgs[1], cmap='gray')
    axes[0, 1].set_title(fila0_titles[1])
    axes[0, 1].axis('off')

    # Las otras dos celdas quedan vacías
    axes[0, 2].axis('off')
    axes[0, 3].axis('off')

    # FILA 1
    for i in range(4):
        axes[1, i].imshow(fila1_imgs[i], cmap='gray')
        axes[1, i].set_title(fila1_titles[i])
        axes[1, i].axis('off')

    # FILA 2
    for i in range(4):
        axes[2, i].imshow(sobel_imgs[i], cmap='gray')
        axes[2, i].set_title(sobel_titles[i])
        axes[2, i].axis('off')

    # FILA 3
    for i in range(4):
        axes[3, i].imshow(prewitt_imgs[i], cmap='gray')
        axes[3, i].set_title(prewitt_titles[i])
        axes[3, i].axis('off')

    # FILA 4
    for i in range(4):
        axes[4, i].imshow(canny_imgs[i], cmap='gray')
        axes[4, i].set_title(canny_titles[i])
        axes[4, i].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()