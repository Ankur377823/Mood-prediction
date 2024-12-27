import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Set page configuration
st.set_page_config(page_title="Mood Prediction", page_icon="😃", layout="wide")

# Load the pre-trained model
MODEL_PATH = "imagepred.keras"  # Ensure this model file is available
try:
    model = load_model(MODEL_PATH)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Define the mood dictionary
mood_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

# Display the title
st.title("Mood Prediction 😃")
st.markdown("Upload a photo of a face, and the model will predict the mood based on facial expression.")

# Upload the image file
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Preprocess the uploaded image
        img = image.load_img(uploaded_file, target_size=(64, 64))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array /= 255.0  # Normalize the image

        # Show the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        # Predict mood when the button is clicked
        if st.button("Predict Mood"):
            with st.spinner("Predicting..."):
                prediction = model.predict(img_array)
                predicted_class_index = np.argmax(prediction, axis=1)[0]
                predicted_mood = mood_dict.get(predicted_class_index, 'Unknown')
                st.markdown(f"**Predicted Mood:** {predicted_mood}")
                st.write("The model has classified the mood based on the facial expression in the image.")
    except Exception as e:
        st.error(f"Error processing the image: {e}")
else:
    st.info("Please upload an image to proceed.")
