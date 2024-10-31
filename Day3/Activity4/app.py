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
import streamlit as st
import warnings
from streamlit_option_menu import option_menu
from streamlit_extras.mention import mention

warnings.filterwarnings("ignore")

st.set_page_config(page_title="News Summarizer Tool", page_icon="", layout="wide")

with st.sidebar :
    #st.image('AI_First_Day_3_Activity_4/images/AI_1.png')
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

   st.title("This is the Home Page!")
   st.write("Introduce Your Chatbot!")
   st.write("What is their Purpose?")
   st.write("What inspired you to make [Chatbot Name]?")
   
elif options == "About Us" :
     st.title("About Us")
     st.write("# [Name]")
     #st.image('images/Meer.png')
     st.write("## [Title]")
     st.text("Connect with me via Linkedin : [LinkedIn Link]")
     st.text("Other Accounts and Business Contacts")
     st.write("\n")

# Options : Model
elif options == "Model" :
    st.title('News Summarizer Tool')
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        News_Article = st.text_input("News Article", placeholder="News : ")
        submit_button = st.button("Generate Summary")

    if submit_button:
        with st.spinner("Generating Summary..."):
            system_prompt = """
You are an expert news summarizer. When given a news article, your task is to condense the information into a brief summary, maintaining accuracy, neutrality, and clarity. Your summary should be 3-5 sentences long and focus on the key facts, events, and figures of the story. Include the essential who, what, where, when, and why, highlighting any significant impacts or next steps. Avoid subjective language, speculative information, and unnecessary background context. Keep the tone objective and concise.
            """
            user_message = f"Here is the news article: {News_Article}"

            struct = [{"role": "system", "content": system_prompt}]

            struct.append({"role": "user", "content": user_message})
            chat = openai.ChatCompletion.create(model="gpt-4o-mini",messages = struct)
            response = chat.choices[0].message.content
            struct.append({"role": "assistant", "content": response})

            st.success("Insight generated successfully!")
            st.subheader("Summary: ")
            st.write(response)
