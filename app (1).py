import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("/content/mnist_digit_recognizer.h5")

st.title("MNIST Digit Recognizer 🔢")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    image = image.resize((28, 28))
    image_array = np.array(image)
    image_array = 255 - image_array  # Invert colors
    image_array = image_array / 255.0
    image_array = image_array.reshape(1, 28, 28, 1)

    st.image(image, caption="Uploaded Image", width=150)
    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction)

    st.write(f"### Predicted Digit: {predicted_class}")




