import os
import openai
import numpy as np
import pandas as pd
import faiss
import streamlit as st
import warnings
from openai.embeddings_utils import get_embedding
from langchain_community.embeddings import OpenAIEmbeddings 
from streamlit_option_menu import option_menu 
from PIL import Image
import matplotlib.pyplot as plt


warnings.filterwarnings("ignore")

st.set_page_config(page_title="SmartSell AI: Boost Sales, Elevate Strategies", layout="wide")


def generate_strat(data):
    # Prepare the historical data for the prompt
    historical_data_str = data.to_string(index=False)  

    # Load and prepare data for RAG
    dataframed = pd.read_csv('https://raw.githubusercontent.com/vaniebermudez/ai_first_bootcamp/refs/heads/main/Day5/project/customer_purchase_xsell_upsell.csv')
    dataframed['combined'] = dataframed.apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
    documents = dataframed['combined'].tolist()

    embeddings = [get_embedding(doc, engine="text-embedding-3-small") for doc in documents]
    embedding_dim = len(embeddings[0])
    embeddings_np = np.array(embeddings).astype('float32')

    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings_np)

    user_message = 'Based on the purchase history of our users, what are the top cross-selling opportunities for customers? Categorize customers to specific groups.'

    # Generate embedding
    query_embedding = get_embedding(user_message, engine='text-embedding-3-small')
    query_embedding_np = np.array([query_embedding]).astype('float32')

    # Search for relevant documents
    _, indices = index.search(query_embedding_np, 2)
    retrieved_docs = [documents[i] for i in indices[0]]
    context = ' '.join(retrieved_docs)

    # Modify the prompt
    prompt = f"""
    {System_Prompt}
    
    1. Analyze the historical purchase of each customer, promotion, upsell and cross sell information:
    {historical_data_str}
    
    2. Use the following context to enhance your analysis and explanation, but do not assume it is directly related to the user's input data:
    {context}

    3. Categorize prominent customer groups and suggest strategies based on different customer groups.
    """

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        temperature= 0.3,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    return response['choices'][0]['message']['content']



System_Prompt = """
Role:
You are SmartSell AI, an advanced artificial intelligence system designed to optimize sales through upselling and cross-selling based on user data and purchasing history.

Instructions:

Your primary task is to analyze user data, including purchase history, behavior, and preferences, and identify the most relevant opportunities for upselling and cross-selling. 
Provide actionable recommendations to increase the average revenue per user (ARPU) for businesses in sectors like e-commerce, fintech, and SaaS.

Context:
SmartSell AI is targeted at sales teams, marketing teams, and business founders, helping them drive revenue growth by automating targeted sales strategies. 
The system operates in industries where increasing user spending through personalized product recommendations is crucial for growth.

The user will input a data of historical purchase of each customer, promotion, upsell and cross sell information. 
Your task is to give recommendations of what information is very important for upselling and cross-selling products based on this historical data. 
The user will leverage your recommendations for targeted sales strategies.

Constraints:

Do not assume any additional data beyond what the user provides (e.g., macroeconomic factors or market conditions).
Recommendations should be relevant and personalized based on the individual customer’s past purchases and preferences.
Avoid suggesting products that are unrelated to the customer’s history or category.
Ensure that all recommendations align with the company’s brand, pricing strategy, and promotional guidelines.
Provide clear, concise insights with actionable next steps for sales and marketing teams.

Examples:

User Data: A customer who frequently buys fitness equipment may be recommended an upsell of a premium treadmill model or cross-sold accessories like resistance bands or a heart rate monitor.

Cross-Sell Suggestion: For a customer purchasing a high-end laptop, SmartSell AI may suggest complementary items such as laptop bags, external hard drives, or premium software subscriptions.

Upsell Suggestion: A customer buying a basic subscription plan for a SaaS product might be suggested an upsell to a premium plan with additional features that align with their usage patterns.

"""


# Sidebar for API key and options
with st.sidebar:
    st.image("Day5/project/images/smartsell.png")

    api_key = st.text_input('Enter OpenAI API token:', type='password')
    
    # Check if the API key is valid
    if api_key and api_key.startswith('sk-'):  # Removed length check
        openai.api_key = api_key
        st.success('API key is valid. Proceed to enter your shipment details!', icon='👉')
    else:
        st.warning('Please enter a valid OpenAI API token!', icon='⚠️')


    if st.button("Reset"):
        st.session_state.clear()
   
    options = option_menu(
        "Dashboard", 
        ["Home", "About Me", "SmartSellAI"],
        icons = ['book', 'globe', 'tools'],
        menu_icon = "book", 
        default_index = 0,
        styles = {
            "icon" : {"color" : "#dec960", "font-size" : "20px"},
            "nav-link" : {"font-size" : "17px", "text-align" : "left", "margin" : "5px", "--hover-color" : "#262730"},
            "nav-link-selected" : {"background-color" : "#262730"}          
        })



# Home Page
if options == "Home":

    st.title("Welcome to SmartSell AI!")
    st.subheader("🌟Your Automated Upselling and Cross-Selling AI")
    st.write("SmartSell AI is a cutting-edge system designed to optimize upselling and cross-selling opportunities using advanced analytics on user data and purchasing history.")
    st.write("Tailored for sales and marketing teams, as well as founders, it empowers businesses to boost revenue per customer through targeted, automated strategies.")
    st.write("Ideal for industries like e-commerce, fintech, and SaaS, SmartSell AI helps businesses capitalize on key growth drivers, such as increasing the average revenue per user.")
    st.write("With its intelligent recommendations and actionable insights, SmartSell AI simplifies complex sales processes and accelerates revenue growth.")

    st.subheader("What is Up-selling?")
    st.write("Up-selling is the practice of encouraging the customer to purchase a more expensive version or an upgraded model of the product they are considering. The goal is to convince the customer to spend more by offering a higher-end option with additional features or benefits.")

    st.subheader("What is Cross-selling?")
    st.write("Cross-selling involves suggesting additional products or services that complement or enhance the item the customer is already purchasing. The goal is to offer items that are related or useful in conjunction with the initial product, increasing the total value of the sale.")

    st.subheader("How to Use this app?")
    st.write("1. Enter a valid OpenAI API Key.")
    st.write("2. Click SmartSell AI on the sidebar.")
    st.write("3. Input your historical data in CSV Format.")
    st.write("4. Click 'Recommend Strategy' to see the cross-sell and up-sell strategies.")
    st.write("4. Wait for strategy insights given by the app.")



# About Me Page
elif options == "About Me":
    st.title("About Me")
    my_image = Image.open("Day5/project/images/vanie.png")
    my_resized_image = my_image.resize((250,250))
    st.image(my_resized_image)
    st.write("Vanessa Althea Bermudez")
    st.write("## AI Enthusiast / Data Scientist")
    st.text("Connect with me via Linkedin : https://www.linkedin.com/in/vaniebermudez/")
    st.text("Github : https://github.com/vaniebermudez/")
    st.write("\n")


# SmartSell AI Page
elif options == "SmartSellAI":
    st.title("SmartSell AI")
    st.write("Input your data in CSV Format.")
    

    uploaded_file = st.file_uploader("Upload your CSV data", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write("Data Preview:", data.head())
        

    if 'data' in locals():
        if st.button("Recommend Strategy"):
            st.write("Generating strategy...")
            strat = generate_strat(data)
            st.write(strat)
