import numpy as np
from sklearn.cluster import KMeans

def segment_image(image, n_clusters) :
    image_shape = image.shape
    flatten_image = np.reshape(image, (-1,3))
    
    KMeans_model = KMeans(n_clusters=n_clusters)
    KMeans_model.fit(flatten_image)
    
    segmented_image = KMeans_model.cluster_centers_[KMeans_model.labels_]
    segmented_image = np.reshape(segmented_image, image_shape).astype(np.uint8)
    
    return segmented_image