"""Streamlit frontend for a chat interface interacting with a database using OpenAI."""

import sys
import os

sys.path.append(os.path.abspath('.'))
#import mysql.connector
import streamlit as st
from sqlalchemy import create_engine
import openai
from llama_index import VectorStoreIndex, SQLDatabase
from llama_index.objects import ObjectIndex, SQLTableNodeMapping, SQLTableSchema
from llama_index.callbacks import CallbackManager, TokenCountingHandler
from llama_index.indices.struct_store import SQLTableRetrieverQueryEngine
from llama_index import ServiceContext
from llama_index.llms import OpenAI
from transformers import  SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import torch
from datasets import load_dataset
import soundfile as sf
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

def load_chain(openai_api_key):
    # Database connection configuration
    db_user = "root"
    db_password = "new_password"
    db_host = "localhost"
    db_name = "caption_database"
    connection_string = f"mysql+pymysql://{db_user}:{db_password}@{db_host}/{db_name}"
    engine = create_engine(connection_string)

    # Initialize SQL database
    sql_database = SQLDatabase(engine)
    tables = list(sql_database._all_tables)
    table_node_mapping = SQLTableNodeMapping(sql_database)
    table_schema_objs = [SQLTableSchema(table_name=table, context_str=table) for table in tables]

    # Token counter and callback manager
    token_counter = TokenCountingHandler()
    callback_manager = CallbackManager([token_counter])

    # OpenAI model and service context
    openai.api_key = openai_api_key
    llm = OpenAI(temperature=0.1, model="gpt-3.5-turbo")
    service_context = ServiceContext.from_defaults(llm=llm, callback_manager=callback_manager)

    # Object index and query engine
    obj_index = ObjectIndex.from_objects(table_schema_objs, table_node_mapping, VectorStoreIndex, service_context=service_context)
    query_engine = SQLTableRetrieverQueryEngine(sql_database, obj_index.as_retriever(similarity_top_k=3), service_context=service_context)
    
    return query_engine


def main():
    # Set page title
    st.set_page_config(page_title="Image Captioning")

    # Sidebar configuration
    openai_api_key = st.sidebar.text_input('Enter your OpenAI API Key and hit Enter', type="password")
    st.write("You can provide your OpenAI API in the sidebar.")
   
    # Chat input
    user_input = st.text_input("Enter your question:", key="input")
    if user_input and openai_api_key:
        # Load the LLM chain
        chain = load_chain(openai_api_key)

        # Display chat messages from history
        if "messages" not in st.session_state:
            st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Process user input
        if user_input:
            with st.chat_message("user"):
                st.markdown(user_input)
            with st.chat_message("assistant"):
                with st.spinner('Chatbot is working...'):
                    assistant_response = chain.query(user_input)
                    st.markdown(assistant_response)
                    
                    print(assistant_response)
                    # Initialize speech synthesis models
                    speech_processor, speech_model, speech_vocoder, speaker_embeddings = initialize_speech_synthesis()

                    if isinstance(assistant_response, list):
                        summary = " ".join(map(str, assistant_response))
                    else:
                        summary = str(assistant_response)

                    # Generate speech from the summary
                    with st.spinner("Generating Speech..."):
                        generate_speech(speech_processor, speech_model, speech_vocoder, speaker_embeddings, summary)

                    # Play the generated sound
                    play_sound()
        
                    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
    else:
        st.write("Please enter Open AI key in the side bar")
                     

if __name__ == "__main__":
    main()
