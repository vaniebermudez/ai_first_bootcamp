import os
import openai
import numpy as np
import pandas as pd
import json
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from openai.embeddings_utils import get_embedding
import faiss
import streamlit as st
import warnings
from streamlit_option_menu import option_menu
from streamlit_extras.mention import mention

warnings.filterwarnings("ignore")

st.set_page_config(page_title="SurePath, Your Financial Guide", page_icon="", layout="wide")

with st.sidebar :
    # st.image('images/White_AI Republic.png')
    openai.api_key = st.text_input('Enter OpenAI API token:', type='password')
    if not (openai.api_key.startswith('sk-') and len(openai.api_key)==164):
        st.warning('Please enter your OpenAI API token!', icon='⚠️')
    else:
        st.success('Proceed to entering your prompt message!', icon='👉')
    with st.container() :
        l, m, r = st.columns((1, 3, 1))
        with l : st.empty()
        with m : st.empty()
        with r : st.empty()

    options = option_menu(
        "Dashboard", 
        ["Home", "About Us", "Model"],
        icons = ['book', 'globe', 'tools'],
        menu_icon = "book", 
        default_index = 0,
        styles = {
            "icon" : {"color" : "#dec960", "font-size" : "20px"},
            "nav-link" : {"font-size" : "17px", "text-align" : "left", "margin" : "5px", "--hover-color" : "#262730"},
            "nav-link-selected" : {"background-color" : "#262730"}          
        })


if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'chat_session' not in st.session_state:
    st.session_state.chat_session = None  # Placeholder for your chat session initialization

# Options : Home
if options == "Home" :

   st.title("Chat with SurePath : Your Personal Financial Guide!")
   st.write("Hello! I'm SurePath, your personal financial guide. I’m here to help you make smart, confident decisions about your insurance and investments. Whether you're just starting out or looking to optimize your financial portfolio, I’ll recommend options that fit your unique goals, risk comfort, and budget. Let’s work together to create a secure, prosperous path toward your future!")
   
elif options == "About Us" :
     st.title("About Us")
     st.write("# Vanessa Althea Bermudez")
    #  st.image('images/vanie.jpg')
     st.write("## AI Enthusiast / Data Scientist")
     st.text("Connect with me via Linkedin : https://www.linkedin.com/in/vaniebermudez/")
     st.text("Github : https://github.com/vaniebermudez/")
     st.write("\n")


elif options == "Model" :
     System_Prompt = """
You are SurePath, a knowledgeable and empathetic AI financial advisor specializing in insurance and investment recommendations. Your goal is to help users find the right products based on their unique needs, risk tolerance, and budget. Approach each conversation with clarity, making complex financial terms easy to understand while offering expert recommendations tailored to the user’s situation.

In your responses:

Gather information on the user’s goals, risk appetite, financial priorities, and budget constraints.
Provide clear, tailored recommendations for insurance or investment products, explaining how each option aligns with their needs and financial goals.
Educate the user on relevant financial concepts when needed, using simple language.
Encourage responsible decision-making by presenting potential risks and benefits, while guiding them toward well-rounded choices.
Stay supportive and neutral, empowering the user to make informed decisions without pressuring them toward specific products.
Always aim to create a positive, reassuring experience that inspires confidence in SurePath as a reliable financial companion.
"""


     def initialize_conversation(prompt):
         if 'message' not in st.session_state:
             st.session_state.message = []
             st.session_state.message.append({"role": "system", "content": System_Prompt})
             chat =  openai.ChatCompletion.create(model = "gpt-4o-mini", messages = st.session_state.message, temperature=0.5, max_tokens=1500, top_p=1, frequency_penalty=0, presence_penalty=0)
             response = chat.choices[0].message.content
             st.session_state.message.append({"role": "assistant", "content": response})

     initialize_conversation(System_Prompt)

     for messages in st.session_state.message :
         if messages['role'] == 'system' : continue 
         else :
            with st.chat_message(messages["role"]):
                 st.markdown(messages["content"])

     if user_message := st.chat_input("Say something"):
        with st.chat_message("user"):
             st.markdown(user_message)
        st.session_state.message.append({"role": "user", "content": user_message})
        chat =  openai.ChatCompletion.create(model = "gpt-4o-mini", messages = st.session_state.message, temperature=0.5, max_tokens=1500, top_p=1, frequency_penalty=0, presence_penalty=0)
        response = chat.choices[0].message.content
        with st.chat_message("assistant"):
             st.markdown(response)
        st.session_state.message.append({"role": "assistant", "content": response})