import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration, SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import torch
from datasets import load_dataset
import soundfile as sf
import easyocr
import cv2
import openai
import pandas as pd
# Download NLTK resources (if needed)
#import nltk
#from nltk.corpus import stopwords
#from nltk.tokenize import word_tokenize
#from nltk.stem import WordNetLemmatizer
#from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.cluster import KMeans
import mysql.connector
from sqlalchemy import create_engine

# Function to initialize BLIP VQA model
@st.cache_resource
def initialize_blip_vqa_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    return processor, model

# Function to initialize speech synthesis models
@st.cache_resource
def initialize_speech_synthesis():
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
    return processor, model, vocoder, speaker_embeddings

def generate_speech(processor, model, vocoder, speaker_embeddings, caption):
    inputs = processor(text=caption, return_tensors="pt")
    speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
    sf.write("speech.wav", speech.numpy(), samplerate=16000)

def play_sound():
    audio_file = open("speech.wav", 'rb')
    audio_bytes = audio_file.read()
    st.audio(audio_bytes, format='audio/wav')

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
    st.title("Image Captioning and OCR with audio output")
    st.write("This app generates descriptive captions for images and text within the image.")

    # Upload image
    uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
   

    
    if uploaded_image is not None:
        # Display uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Initialize BLIP VQA model
        processor, model = initialize_blip_vqa_model()

        # Input question
        #question = st.text_input("Enter a question about the image")

        # Generate caption
        with st.spinner("Generating caption..."):
            inputs = processor(image, return_tensors="pt")
            output = model.generate(**inputs)
            caption = processor.decode(output[0], skip_special_tokens=True)

        # Display caption
        st.subheader("Caption:")
        st.write(caption)
        reader = easyocr.Reader(['en']) 
        result = reader.readtext(image)
        box_list = []
        # Print the detected text and its bounding boxes
        for detection in result:
            text, box, score = detection
            print(f'Text in the image:{box}')
            box_list.append(box) 
        st.write("caption with text")
        combined_text = f" {caption}\n: {box_list}"
        st.write(combined_text)
        # Answer question
        
       

        text = "caption"
        summary = combined_text

        insert_caption_data(text, summary)

       
       
        # Initialize speech synthesis models
        speech_processor, speech_model, speech_vocoder, speaker_embeddings = initialize_speech_synthesis()
        # Generate speech from the caption
        with st.spinner("Generating Speech..."):
            generate_speech(speech_processor, speech_model, speech_vocoder, speaker_embeddings, combined_text)
        #st.write(generate_speech(speech_processor, speech_model, speech_vocoder, speaker_embeddings, output_caption))
        st.subheader("Audio:")
        # Play the generated sound
        play_sound()

 
    

if __name__ == "__main__":
    main()
