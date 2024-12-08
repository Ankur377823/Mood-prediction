import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import base64
from io import BytesIO

# Mood dictionary for class mapping
mood_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

# Load the pre-trained model
model = load_model('image.keras')  # Ensure path is correct

# Streamlit app layout with custom page config
st.set_page_config(page_title="Mood Prediction", page_icon="😃", layout="wide")

# Apply custom CSS for a more stylish interface
st.markdown(
    """
    <style>
    body {
        background-color: #f5f5f5;
        font-family: 'Helvetica', sans-serif;
    }
    .title {
        color: #FF6347;
        font-size: 40px;
        text-align: center;
    }
    .description {
        color: #333;
        font-size: 18px;
        text-align: center;
        margin-bottom: 30px;
    }
    .button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 20px;
        font-size: 16px;
        border-radius: 5px;
        border: none;
        cursor: pointer;
    }
    .button:hover {
        background-color: #45a049;
    }
    .image {
        border-radius: 10px;
        border: 5px solid #ddd;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
        width: 100%;
        max-width: 400px;
        margin: 0 auto;
    }
    .prediction {
        font-size: 24px;
        color: #2c3e50;
        text-align: center;
        margin-top: 20px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True
)

# Header
st.markdown("<h1 class='title'>Mood Prediction 😃</h1>", unsafe_allow_html=True)

# Description
st.markdown("<p class='description'>Upload a photo of a face to predict the mood based on facial expression. The model classifies the mood into one of these categories:</p>", unsafe_allow_html=True)

# List of moods
st.write("**Moods**: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise")

# File uploader for image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Open the image and process it for the model (resize to 64x64x3)
    img = image.load_img(uploaded_file, target_size=(64, 64))  # Resize to 64x64
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize the image if needed

    # Convert the uploaded file to base64
    buffered = BytesIO(uploaded_file.read())
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # Display the uploaded image with custom styling using st.markdown
    st.markdown(
        f'<img src="data:image/jpeg;base64,{img_base64}" class="image"/>',
        unsafe_allow_html=True
    )

    # Prediction Button
    if st.button('Predict Mood', key="predict", help="Click to predict the mood from the image.", use_container_width=True):
        # Predict the class
        prediction = model.predict(img_array)

        # Get predicted class index
        predicted_class_index = np.argmax(prediction, axis=1)[0]  # Get the class index

        # Map the index to the corresponding mood
        predicted_mood = mood_dict.get(predicted_class_index, 'Unknown')  # Default to 'Unknown' if index is not in the dict

        # Display prediction result
        st.markdown(f"<div class='prediction'>Predicted Mood: {predicted_mood}</div>", unsafe_allow_html=True)

        # Display additional message
        st.write("The model has classified the mood based on the facial expression present in the image.")
