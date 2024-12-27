import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np


st.set_page_config(page_title="Mood Prediction", page_icon="😃", layout="wide")


MODEL_PATH = "imagepred.keras"
try:
    model = load_model(MODEL_PATH)
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()


mood_dict = {0:'Angry',1:'Disgust',2:'Fear',3:'Happy',4:'Neutral',5:'Sad',6:'Surprise'}


def preprocess_image(uploaded_image):
    try:
        img = image.load_img(uploaded_image, target_size=(64, 64))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  
        img_array /= 255.0  
        return img_array
    except Exception as e:
        st.error(f"Error processing the image:{e}")
        return None


def predict_mood(img_array):
    try:
        prediction = model.predict(img_array)
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        predicted_mood = mood_dict.get(predicted_class_index, 'Unknown')
        return predicted_mood
    except Exception as e:
        st.error(f"Error during prediction:{e}")
        return None


st.title("Mood Prediction 😃")
st.markdown(
    """
    Upload a photo of a face, and the model will predict the mood based on facial expression.
    """
)


st.subheader("The model can identify the following expressions:")
for key, value in mood_dict.items():
    st.markdown(f"- **{value}**")


uploaded_file = st.file_uploader("Choose an image file (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    col1, col2 = st.columns(2)

   
    with col1:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

   
    img_array = preprocess_image(uploaded_file)
    if img_array is not None and st.button("Predict Mood"):
        with st.spinner("Analyzing..."):
            predicted_mood = predict_mood(img_array)

      
        with col2:
            if predicted_mood:
                st.markdown(f"### **Predicted Mood:** {predicted_mood}")
                st.write("The model has classified the mood based on the facial expression.")
            else:
                st.error("Could not predict the mood. Please try again with a different image.")
else:
    st.info("Upload an image to start the mood prediction.")
