import matplotlib.pyplot as plt
import cv2
import numpy as np
from sklearn.cluster import KMeans

image = cv2.imread('test_1.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

image_shape = image.shape

flatten_image = image.reshape(-1, 3)

number_of_clusters = 2
KMeans_model = KMeans(n_clusters=number_of_clusters)
KMeans_model.fit(flatten_image)

image_segmentation = KMeans_model.cluster_centers_[KMeans_model.labels_]

segmented_image = image_segmentation.reshape(image_shape)

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
fig.suptitle("Image segmentation using KMeans")

ax[0].imshow(image)
ax[0].set_title('Original Image')
ax[0].axis('off')

ax[1].imshow(segmented_image.astype(np.uint8))
ax[1].set_title('Segmented Image')
ax[1].axis('off')

plt.tight_layout()
plt.show()
