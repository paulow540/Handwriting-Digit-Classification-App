import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("handwritten_model.h5")


def preprocess_image(image):
    
    # Convert to grayscale
    image = image.convert('L')

    # Resize to 28x28
    image = image.resize((28, 28))

    # Convert to numpy array
    image = np.array(image)

    # Invert colors (important for MNIST)
    image = 255 - image

    # Normalize
    image = image / 255.0

    return image


def main():

    st.title("Handwriting Digit Classification App")

    st.write("Upload a handwritten digit image (0-9)")

    image_file = st.file_uploader(
        "Upload Image",
        type=["png", "jpg", "jpeg"]
    )

    if image_file is not None:

        image = Image.open(image_file)

        st.image(image, caption="Uploaded Image", width=200)

        processed_image = preprocess_image(image)

        st.write("Processed Image")

        st.image(processed_image, width=150)

        if st.button("Predict"):

            prediction = model.predict(
                np.expand_dims(processed_image, axis=0)
            )

            st.write(prediction)  

            predicted_class = np.argmax(prediction)

            confidence = np.max(prediction)

            st.success(f"Prediction: {predicted_class}")

            st.info(f"Confidence: {confidence:.2f}")


if __name__ == "__main__":
    main()