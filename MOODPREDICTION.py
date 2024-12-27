import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Set page configuration (must be the first streamlit command)
st.set_page_config(page_title="Mood Prediction", page_icon="😃", layout="wide")

# Load the pre-trained model (ensure the path is correct)
MODEL_PATH = "image_new.keras"  # Ensure this model file is available on Streamlit Cloud
try:
    model = load_model("imagepred.keras")
except Exception as e:
    st.error(f"Error loading model: {e}")

# Define the mood dictionary
mood_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

# Display the title
st.title("Mood Prediction 😃")
st.markdown("Upload a photo of a face, and the model will predict the mood based on facial expression.")

# Upload the image file
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Preprocess the uploaded image
    img = image.load_img(uploaded_file, target_size=(64, 64))  # Resize to match model input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize the image

    # Show the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Make prediction when the button is clicked
    if st.button("Predict Mood"):
        prediction = model.predict(img_array)
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        predicted_mood = mood_dict.get(predicted_class_index, 'Unknown')  # Default to 'Unknown' if index not in dict
        st.markdown(f"**Predicted Mood:** {predicted_mood}")
        st.write("The model has classified the mood based on the facial expression in the image.")
