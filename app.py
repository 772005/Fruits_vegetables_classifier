# import issential library and packages 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import  load_model
import streamlit as st
import numpy as np 
from PIL import Image


#-----------App configuration--------------

st.set_page_config(page_title='Fruit and Vegetable Classifier',
                   page_icon='🍎',
                   layout='centered')


st.markdown(
    '''
    <style>
    .stApp {
        background-image: url("background.jpg");
        background-size: cover;
        background-position: center;
        background-color: black;
    }
    h1, h2, h3 {
        text-align: center;
    }
    .stImage img {
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
    </style>
    <h1 style="color: #4CAF50; text-align: center;">Fruit and Vegetable Classifier</h1>
    <p style="color: #FFFFFF; text-align: center;">Upload an image and let the AI classify it!</p>
    <p style="color: #FFFFFF; text-align: center;">If it's not in the trained model list, it will be marked as <em>Unknown❓🤷</em></p>
    <br>
    '''
    , unsafe_allow_html=True
)
#------------------- Load Model -------------------#
# @st.cache_resource
# def load_classification_model():
#     return load_model(
#         'C:\\Users\\Lenovo\\Desktop\\project\\fruit_vegi_classifier\\Fruits_vegetables_classifier.keras',
#          compile=False
#     )

# model = load_classification_model()

@st.cache_resource
def load_classification_model():
    # Use relative path so it works both locally and on Streamlit Cloud
    model_path = os.path.join(os.path.dirname(__file__), "Fruits_vegetables_classifier.keras")
    return load_model(model_path, compile=False)

model = load_classification_model()

#-------------------- Categories --------------------#
data_cat = [
 'apple',
 'banana',
 'beetroot',
 'bell pepper',
 'cabbage',
 'capsicum',
 'carrot',
 'cauliflower',
 'chilli pepper',
 'corn',
 'cucumber',
 'eggplant',
 'garlic',
 'ginger',
 'grapes',
 'jalepeno',
 'kiwi',
 'lemon',
 'lettuce',
 'mango',
 'onion',
 'orange',
 'paprika',
 'pear',
 'peas',
 'pineapple',
 'pomegranate',
 'potato',
 'raddish',
 'soy beans',
 'spinach',
 'sweetcorn',
 'sweetpotato',
 'tomato',
 'turnip',
 'watermelon']


img_height , img_width = 180, 180
threshold = 0.60

#---------------------Input selection---------------------#
option = st.radio("Select Input Method:", ["Upload Image", "Capture Live Image"])

def preprocess_image(img: Image.Image):
    img = img.convert('RGB').resize((img_height, img_width))
    img_arr = tf.keras.utils.img_to_array(img)
    img_batch = tf.expand_dims(img_arr, axis=0)
    return img_batch

#---------------------Uploaded Image---------------------#
if option == "Upload Image":
    uploaded_file = st.file_uploader(
        "📂 Choose an image...",
        type=['jpg', 'jpeg', 'png']
    )
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        img_batch = preprocess_image(img)
        predict = model.predict(img_batch)
        score = tf.nn.softmax(predict)
        max_score = np.max(score)

        st.image(img, caption="Uploaded Image", width=250)

        if max_score < threshold:
            st.subheader("🤷 Unknown Object")
            st.write("This doesn't seem to be a fruit or vegetable.")
            st.write(f"Confidence: *{max_score * 100:.2f}%*")
        else:
            st.subheader(f"Prediction: *{data_cat[np.argmax(score)]}*")
            st.write(f"Confidence: *{max_score * 100:.2f}%*")
    else:
        st.info("Please upload an image file to start classification.")

#---------------------Live Capture Image---------------------#
else:
    captured_image = st.camera_input("📸 Capture Image")
    if captured_image is not None:
        img = Image.open(captured_image)
        img_batch = preprocess_image(img)
        predict = model.predict(img_batch)
        score = tf.nn.softmax(predict)
        max_score = np.max(score)

        st.image(img, caption="Captured Image", width=250)

        if max_score < threshold:
            st.subheader("🤷 Unknown Object")
            st.write("This doesn't seem to be a fruit or vegetable.")
            st.write(f"Confidence: *{max_score * 100:.2f}%*")
        else:
            st.subheader(f"Prediction: *{data_cat[np.argmax(score)]}*")
            st.write(f"Confidence: *{max_score * 100:.2f}%*")
    else:
        st.info("Please capture an image to start classification.")
