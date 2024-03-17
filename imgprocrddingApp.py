import streamlit as st
import cv2
import numpy as np

def rgb_to_hsv(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

def segment_image_by_intensity(image, intensity_threshold):
    hsv_image = rgb_to_hsv(image)
    v_channel = hsv_image[:, :, 2]
    mask = np.where(v_channel > intensity_threshold, 255, 0).astype(np.uint8)
    segmented_image = cv2.bitwise_and(image, image, mask=mask)
    return segmented_image

def reconstruct_image_from_edge_map(edge_map):
    mask = np.where(edge_map > 0, 0, 255).astype(np.uint8)
    reconstructed_image = cv2.inpaint(edge_map, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return reconstructed_image

def main():
    st.title("Image Processing App")

    # File uploader for image selection
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        # Display the uploaded image
        image = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), 1)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Convert RGB to HSV
        hsv_button = st.button("Convert to HSV")
        if hsv_button:
            hsv_image = rgb_to_hsv(image)
            st.image(hsv_image, caption="HSV Image", use_column_width=True)

        # Segment image based on intensity
        intensity_threshold = st.slider("Intensity Threshold", min_value=0, max_value=255, value=100, step=1)
        segment_button = st.button("Segment Image")
        if segment_button:
            segmented_image = segment_image_by_intensity(image, intensity_threshold)
            st.image(segmented_image, caption="Segmented Image", use_column_width=True)

        # Reconstruct image from edge map
        edge_map_button = st.button("Reconstruct Image from Edge Map")
        if edge_map_button:
            reconstructed_image = reconstruct_image_from_edge_map(segmented_image)
            st.image(reconstructed_image, caption="Reconstructed Image", use_column_width=True)

if __name__ == "__main__":
    main()
