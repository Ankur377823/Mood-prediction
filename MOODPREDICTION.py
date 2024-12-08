import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Mood dictionary for class mapping
mood_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

# Load the pre-trained model
model = load_model('image.keras')  # Ensure path is correct

# Streamlit app layout
st.title('CNN Model Deployment')
st.write('Upload an image for prediction')

# File uploader for image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Open the image and process it for the model (resize to 64x64x3)
    img = image.load_img(uploaded_file, target_size=(64, 64))  # Resize to 64x64
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize the image if needed

    # Predict the class
    prediction = model.predict(img_array)

    # Get predicted class index
    predicted_class_index = np.argmax(prediction, axis=1)[0]  # Get the class index

    # Map the index to the corresponding mood
    predicted_mood = mood_dict.get(predicted_class_index, 'Unknown')  # Default to 'Unknown' if index is not in the dict

    # Display the uploaded image and prediction result
    st.image(img, caption='Uploaded Image', use_column_width=True)
    st.write(f'Predicted Mood: {predicted_mood}')
