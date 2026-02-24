import cv2
import numpy as np
import os


train_dir = 'C:/Users/Marce/Documents/SenalesImagenes/Clase3/proyecto/dataset/train'
images = os.listdir(train_dir)

n_clusters = 5
all_pixels = []

for file_name in images:
    img_path = os.path.join(train_dir, file_name)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        all_pixels.append(img.flatten())

pixel_values = np.concatenate(all_pixels).reshape(-1,1).astype(np.float32)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
_, labels_global, centers_global = cv2.kmeans(
    pixel_values, n_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
)

centers_global = centers_global.flatten()


cluster_stats = []

for i in range(n_clusters):
    cluster_pixels = pixel_values[labels_global.flatten() == i]

    mean_val = np.mean(cluster_pixels)
    min_val = np.min(cluster_pixels)
    max_val = np.max(cluster_pixels)

    cluster_stats.append((i, mean_val, min_val, max_val))

# Ordenar por intensidad promedio
cluster_stats_sorted = sorted(cluster_stats, key=lambda x: x[1])

# Cluster del objeto (el intermedio)
objeto = cluster_stats_sorted[len(cluster_stats_sorted)//2][0]


print("\n=== RESULTADOS DEL ENTRENAMIENTO K-MEANS ===")
print("Cluster | Promedio | Mínimo | Máximo")
print("--------------------------------------")

for cid, mean_val, min_val, max_val in cluster_stats_sorted:
    print(f"{cid:<7} | {mean_val:8.2f} | {min_val:6.0f} | {max_val:6.0f}")

print("\nClusters ordenados (oscuro → brillante):")
print([cid for cid,_,_,_ in cluster_stats_sorted])

print(f"\nCluster seleccionado como OBJETO: {objeto}")