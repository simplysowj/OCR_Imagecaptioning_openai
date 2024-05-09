import streamlit as st
from PIL import Image
import requests
from io import BytesIO
from gtts import gTTS
import os

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO

import streamlit as st
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch
from PIL import Image
from nltk.translate.bleu_score import sentence_bleu

st.set_page_config(page_title='Template' ,layout="wide",page_icon='👧🏻')
def local_css(file_name):
    with open(file_name) as f:
        st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)
local_css(r"style\style.css")
st.image("images/1.png", use_column_width=True, caption="The output is not accurate so implemented other models like VGG16 and BLIP")

st.image("images/VIT-GP2.png", use_column_width=True, caption="The output is not accurate so implemented other models like VGG16 and BLIP")
# Load the model and tokenizer
model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}



st.sidebar.markdown("---")
st.sidebar.markdown("Developed by Sowjanya Bojja")
st.sidebar.markdown("Contact: [simplysowj@gmail.com](mailto:simplysowj@gmail.com)")
st.sidebar.markdown("GitHub: [Repo](https://github.com/simplysowj)")

# Function to generate caption
def generate_caption(image):
    pixel_values = feature_extractor(images=[image], return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds[0] if preds else "Caption could not be generated."







# Function to convert text to speech
def text_to_speech(text, lang='en'):
    tts = gTTS(text=text, lang=lang)
    tts.save("output.mp3")
def transform_to_paint(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Invert the grayscale image
    inverted_gray_image = 255 - gray_image
    # Apply bilateral filter to the inverted grayscale image
    bilateral_filtered_image = cv2.bilateralFilter(inverted_gray_image, 11, 17, 17)
    # Invert the bilateral filtered image
    inverted_filtered_image = 255 - bilateral_filtered_image
    # Create the painting effect by blending the inverted filtered image with the original image
    painting_image = cv2.bitwise_and(image, image, mask=inverted_filtered_image)
    return painting_image

def transform_to_sketch(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Invert the grayscale image
    inverted_gray_image = 255 - gray_image
    # Apply Gaussian blur to the inverted image
    blurred_image = cv2.GaussianBlur(inverted_gray_image, (21, 21), 0)
    # Invert the blurred image
    inverted_blurred_image = 255 - blurred_image
    # Create the pencil sketch image by blending the inverted blurred image with the grayscale image
    pencil_sketch_image = cv2.divide(gray_image, inverted_blurred_image, scale=256.0)
    return pencil_sketch_image



def main():
    st.title("Image Processing with NLP")

    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Generate caption for the image
        caption = generate_caption(image)
        st.write("### Image Caption:")
        st.write(caption)

        

        # Convert caption to speech
        text_to_speech(caption)

        # Play audio
        st.audio("output.mp3")


        

if __name__ == '__main__':
    main()
