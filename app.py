import streamlit as st
from PIL import Image
import numpy as np
from kmeans_segmentation import segment_image

st.set_page_config(page_title="KMeans Segmentation", layout="wide")

st.title("Image Segmentation Using K-Means Clustering")

img = st.file_uploader("Upload your image", type=['png', 'jpg', 'jpeg'], label_visibility='visible')

number_of_segments = st.number_input(
    'Select number of segment', min_value=1, max_value=25, value=2, label_visibility='visible')

col1, col2 = st.columns(2)

if img : 
    with col1:
        st.header("Original Image")
        st.image(img)

if img and number_of_segments > 0:
    img = Image.open(img)
    img = np.asarray(img)
    with col2:
        st.header("Segmented Image")
        st.image(segment_image(img, number_of_segments))
