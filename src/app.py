import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import cv2
from model import EnsembledModel

if 'model' not in st.session_state:
    with st.spinner("Loading Model...🤖"):
        st.session_state['model'] = EnsembledModel()

model = st.session_state['model'] 
# Title and description
st.title("Handwritten Digit Recognition 🔢")
st.subheader("University Project 🏫")
st.write("Author : Ouassim HAMDANI")
st.write("Class : Master IIA 1 - Extraction data")

st.write("Draw a digit in the box below:")
col1, col2 = st.columns(2)

# Create a canvas component
with col1:
    canvas_result = st_canvas(
        fill_color="black",  # Fixed background color
        stroke_width=20,
        stroke_color="white",
        background_color="black",
        update_streamlit=True,
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

# Resize & Normalize
if canvas_result.image_data is not None:
    image = canvas_result.image_data
    gray_image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY)

    # Resize the image to 28x28
    resized_image = cv2.resize(gray_image, (32, 32))

    # Normalize pixel values to be between 0 and 1
    normalized_image = resized_image / 255.0


#Predict
if st.button("Predict"):
    if canvas_result.image_data is not None:
        # Reshape the image for the model
        input_data = normalized_image.reshape(1, 32, 32, 1)
        with col2:
            with st.spinner("Predicting...✍️"):
                label = model.predict(input_data,return_labels=True)
            
            
                st.write(f"# {label[0]}")
    else:
        with col2:
            st.write("Please draw a digit first.")
        
