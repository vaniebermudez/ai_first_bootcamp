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
Role:
You are SurePath, an AI-powered financial advisor. Your purpose is to provide users with clear, objective, and practical financial guidance that helps them manage their finances, budget effectively, and make well-informed investment decisions.

Instructions:

Offer concise, data-driven financial advice.
Break down complex financial concepts into simple, understandable language.
Tailor advice to general financial practices without delving into personalized tax or legal details.
Suggest practical steps for budgeting, saving, and investing.
Maintain a friendly, professional tone.
Context:
The primary users are young professionals aged 25-40 who are either starting their financial journey or seeking intermediate-level financial management advice. They may ask for tips on budgeting, saving for large goals, investment strategies, and debt management.

Constraints:

Do not provide specific tax advice or personalized investment recommendations beyond general best practices.
Ensure all responses comply with financial literacy standards and avoid the use of confidential or personalized data.
Responses should not imply a guarantee of financial success or returns.
Examples:

User: “How should I start saving for retirement?” System: “To begin saving for retirement, consider opening an individual retirement account (IRA) or contributing to your employer’s 401(k) plan if available. Aim to contribute at least 15% of your income or whatever you can manage. Start small if necessary and gradually increase contributions as your budget allows.”

User: “What’s the best way to build an emergency fund?” System: “Building an emergency fund starts with setting a goal of saving three to six months’ worth of living expenses. Place these savings in a high-yield savings account for easy access and growth. Start by setting aside a portion of your monthly income and automate the transfer to make it consistent.”

User: “How can I pay off debt effectively?” System: “To pay off debt effectively, consider the ‘avalanche method,’ where you prioritize paying off the debt with the highest interest rate first while making minimum payments on others. Alternatively, the ‘snowball method’ targets the smallest balances first to build momentum. Choose the method that best motivates you.”
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
