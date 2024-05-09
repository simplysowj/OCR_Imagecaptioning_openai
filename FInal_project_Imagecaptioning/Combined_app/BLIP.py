import streamlit as st
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import os
import torch
import soundfile as sf
from datasets import load_dataset
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image
import requests
from io import BytesIO
from gtts import gTTS
import os
#from googletrans import Translator
import streamlit as st
import cv2
import numpy as np
import easyocr

import mysql.connector

from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import torch

from nltk.translate.bleu_score import sentence_bleu
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Model Description
model_description = """
This application utilizes image captioning and text-to-speech models to generate a caption for an uploaded image 
and convert the caption into speech.


"""


@st.cache_resource
def initialize_image_captioning():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    return processor, model

@st.cache_resource
def initialize_speech_synthesis():
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
    return processor, model, vocoder, speaker_embeddings

def generate_caption(processor, model, image):
    inputs = processor(image, return_tensors="pt")
    out = model.generate(**inputs)
    output_caption = processor.decode(out[0], skip_special_tokens=True)
    return output_caption

def generate_speech(processor, model, vocoder, speaker_embeddings, caption):
    inputs = processor(text=caption, return_tensors="pt")
    speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
    sf.write("speech.wav", speech.numpy(), samplerate=16000)


def play_sound():
    audio_file = open("speech.wav", 'rb')
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format='audio/wav')

def visualize_speech():
    data, samplerate = sf.read("speech.wav")
    duration = len(data) / samplerate

    # Create time axis
    time = np.linspace(0., duration, len(data))

     # Plot the speech waveform
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(time, data)
    ax.set(xlabel="Time (s)", ylabel="Amplitude", title="Speech Waveform")

    # Display the plot using st.pyplot()
    st.pyplot(fig)

# Function to insert data into the "caption" table
def insert_caption_data(text, summary):
    try:
        # Connect to MySQL database
        connection = mysql.connector.connect(
            host="localhost",
            port="3306",
            user="root",
            password="new_password",
            database="caption_database"  # Name of your database
        )

        # Create a cursor object to execute SQL queries
        cursor = connection.cursor()

        # SQL query to insert data into the "caption" table
        sql_insert_query = "INSERT INTO caption (text, summary) VALUES (%s, %s)"
        data = (text, summary)

        # Execute the SQL query
        cursor.execute(sql_insert_query, data)

        # Commit the transaction
        connection.commit()

        # Close cursor and connection
        cursor.close()
        connection.close()

        st.success("Data inserted successfully into the 'caption' table!")
    except mysql.connector.Error as error:
        st.error(f"Error inserting data into 'caption' table: {error}")

def main():
    st.set_page_config(
    page_title="Image-to-Speech",
    page_icon="📸",
    initial_sidebar_state="collapsed",
    menu_items={
        
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)

    
    st.sidebar.markdown("---")
    st.sidebar.markdown("Developed by Sowjanya")
    st.sidebar.markdown("Contact: [simplysowj@gmai.com](mailto:simplysowj@gmai.com)")
    st.sidebar.markdown("GitHub: [Repo](https://github.com/simplysowj)")
    # Add CSS styling to the sidebar
    st.sidebar.markdown(
        """
        <style>
        /* Style the sidebar itself */
        [data-testid=stSidebar] {
            background-image: linear-gradient(#000395, #FFD4DD);
        }

        /* Style hyperlinks */
        .sidebar-content a {
            display: block;
            color: #2e9aff !important;
            font-weight: bold;
        }

        /* Style paragraphs */
        .sidebar-content p {
            color: white;
        }

        /* Customize caret color */
        .st-ck {
            caret-color: black;
        }

        /* Set text color for certain elements */
        .st-bh, .st-c2, .st-c3, .st-c4, .st-c5, .st-c6, .st-c7, .st-c8, .st-c9, .st-ca, .st-cb, .st-b8, .st-cc, .st-cd, .st-ce, .st-cf, .st-cg, .st-ch, .st-ci, .st-cj, .st-ae, .st-af, .st-ag, .st-ck, .st-ai, .st-aj, .st-c1, .st-cl, .st-cm, .st-cn {
            color: black;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <style>
        .container {
            max-width: 800px;
        }
        .title {
            text-align: center;
            font-size: 32px;
            font-weight: bold;
            margin-bottom: 20px;
        }
        .description {
            margin-bottom: 30px;
        }
        .instructions {
            margin-bottom: 20px;
            padding: 10px;
            background-color: #f5f5f5;
            border-radius: 5px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Title
    st.markdown("<div class='title'>Image Captioning with OCR</div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,2,1])

    with col1:
        st.write("")

    with col2:
        st.image("images/blip.png", use_column_width=True, caption="Welcome to my page")

    with col3:
        st.write("")

    # Model Description
    st.markdown("<div class='description'>" + model_description + "</div>", unsafe_allow_html=True)

    # Instructions
    with st.expander("Instructions"):
        st.markdown("1. Upload an image or provide the URL of an image.")
        st.markdown("2. Click the 'Generate Caption and Speech' button.")
        st.markdown("3. The generated caption will be displayed, and the speech will start playing.")


    # Choose image source
    image_source = st.radio("Select Image Source:", ("Upload Image", "Open from URL"))

    image = None

    if image_source == "Upload Image":
        # File uploader for image
        uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
        else:
            image = None

    else:
        # Input box for image URL
        url = st.text_input("Enter the image URL:")
        if url:
            try:
                response = requests.get(url, stream=True)
                if response.status_code == 200:
                    image = Image.open(response.raw)
                else:
                    st.error("Error loading image from URL.")
                    image = None
            except requests.exceptions.RequestException as e:
                st.error(f"Error loading image from URL: {e}")
                image = None

    # Generate caption and play sound button
    if image is not None:
        # Display the uploaded image
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Initialize image captioning models
        caption_processor, caption_model = initialize_image_captioning()

        # Initialize speech synthesis models
        speech_processor, speech_model, speech_vocoder, speaker_embeddings = initialize_speech_synthesis()

        # Generate caption
        with st.spinner("Generating Caption..."):
            output_caption = generate_caption(caption_processor, caption_model, image)

        # Display the caption
        st.subheader("Caption:")
        st.write(output_caption)
        reader = easyocr.Reader(['en']) 
        result = reader.readtext(image)
        box_list = []
        # Print the detected text and its bounding boxes
        for detection in result:
            text, box, score = detection
            print(f'Text in the image:{box}')
            box_list.append(box) 
        st.write("caption with text")
        combined_text = f" {output_caption}\n: {box_list}"
        st.write(combined_text)
        # Answer question
        
       

        text = "caption"
        summary = combined_text

        insert_caption_data(text, summary)
        
        

if __name__ == "__main__":
    main()