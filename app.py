import streamlit as st
from image_pred import detction_from_image
from PIL import Image
import io
from prediction import perform_real_time_perdiction
import cv2
import numpy as np

st.title("Night Jasmine Prediciton using YOLO")

option = st.radio("Choose input source:", ("Upload File", "Camera"), index=None)

if option == "Camera":
    cap = st.camera_input(label="Input From Camera")
    if cap:
        # Convert Streamlit's camera input to OpenCV format
        bytes_data = cap.getvalue()
        np_array = np.frombuffer(bytes_data, np.uint8)
        frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
        perform_real_time_perdiction(cap=frame)
    
    

elif option == "Upload File":

    uploaded_image = st.file_uploader(
        label="Upload Single Image that Contains Night Jasmine!",
        accept_multiple_files=False,
        type=["jpg", "png"],
        label_visibility="visible",
    )
    if uploaded_image is not None:
        with st.spinner("Performing Detection..."):

            bytes_data = uploaded_image.read()

            # saving an image object from the bytes data
            img = Image.open(io.BytesIO(bytes_data))

            # saving the image in jpg format
            img.save("uploaded_image.jpg")
            detected_image = detction_from_image("uploaded_image.jpg")
            col1, col2 = st.columns(2)

            
            with col1:
                st.image(
                    uploaded_image,
                    caption="User Uploaded Image",
                    use_container_width="auto",
                )

            with col2:
                st.image(detected_image, caption="Predictions from the model!")
