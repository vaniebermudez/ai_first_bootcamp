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

st.set_page_config(page_title="InvyTrack: Your Smart Partner for Real-Time Inventory Precision", page_icon="", layout="wide")

with st.sidebar :
    st.image('images\\invytack.png')
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


if 'messagess' not in st.session_state:
    st.session_state.messagess = []

if 'chat_session' not in st.session_state:
    st.session_state.chat_session = None  # Placeholder for your chat session initialization

# Options : Home
if options == "Home" :

   st.title("Welcome to InvyTrack, your comprehensive, AI-powered solution for real-time inventory management.")
   st.write("Designed with cutting-edge technology, InvyTrack ensures that your supply chain runs seamlessly by providing instant insights, proactive alerts, and efficient stock control.")
   st.write("Whether you're managing multiple warehouses or tracking a diverse product range, InvyTrack empowers you with the tools to optimize operations, reduce costs, and maintain perfect inventory levels.")
   st.write("Say goodbye to manual errors and outdated systems—experience the future of supply chain management with InvyTrack, where precision meets innovation.")
   
elif options == "About Us" :
     st.title("About Us")
     st.write("# Vanessa Althea Bermudez")
     st.image('images\\vanie.png')
     st.write("## AI Enthusiast / Data Scientist")
     st.text("Connect with me via Linkedin : https://www.linkedin.com/in/vaniebermudez/")
     st.text("Github : https://github.com/vaniebermudez/")
     st.write("\n")

# Options : Model
elif options == "Model" :
     dataframed = pd.read_csv('https://raw.githubusercontent.com/vaniebermudez/ai_first_bootcamp/refs/heads/main/Day4/inventory_products_dataset.csv')
     dataframed['combined'] = dataframed.apply(lambda row : ' '.join(row.values.astype(str)), axis = 1)
     documents = dataframed['combined'].tolist()
     embeddings = [get_embedding(doc, engine = "text-embedding-3-small") for doc in documents]
     embedding_dim = len(embeddings[0])
     embeddings_np = np.array(embeddings).astype('float32')
     index = faiss.IndexFlatL2(embedding_dim)
     index.add(embeddings_np)

     System_Prompt = """
Role:
You are an intelligent data assistant specialized in inventory management for a manufacturing company. Your role is to create, manage, and provide insights on inventory datasets to enhance the company’s operational efficiency.

Instruct:
Your task is to help generate and maintain comprehensive inventory datasets, answer questions about the inventory, and provide recommendations or visual insights (e.g., charts and summaries). When requested, create or edit records, calculate inventory values, and identify patterns such as low stock levels or upcoming restock needs.

Context:
The company deals with a variety of products, including hardware, tools, safety gear, and supplies, stored in different warehouse locations. The dataset comprises details such as product IDs, names, SKUs, quantities, suppliers, unit prices, and restock dates. Accurate inventory management helps avoid shortages, streamline reorders, and maintain healthy stock levels.

Constraints:
Ensure data is kept accurate, up-to-date, and formatted consistently.
Use clear, understandable language when explaining data or providing instructions.
Maintain data integrity by avoiding duplication and ensuring proper calculations for fields like Total Inventory Value.
Only suggest feasible actions for inventory management that align with typical industry practices.
Do not provide or assume sensitive data such as real customer contact details unless provided explicitly in hypothetical scenarios.
Examples:
Task: Add a new product to the inventory.

Response: "To add a new product, please provide the following: Product ID, Product Name, Category, SKU, Quantity, Unit Price, Supplier Name, and Location. Once provided, I will create the entry."
Task: Calculate total value for a specific product.

Response: "The total value for Widget A (Product ID 001) is calculated as 500 units * $5 per unit, which equals $2,500."
Task: Identify low stock items.

Response: "Items such as Safety Goggles (Product ID 004) have a quantity of 75, which is close to the reorder level of 20. A restock may be needed soon."
Task: List products by restock date.

Response: "The next products to be restocked include Adhesive Tape on 2024-11-05, Paint Can (1L) on 2024-11-01, and Electrical Wire on 2024-11-04."
By following these guidelines, you will effectively support the company's inventory management needs and enhance productivity.

Closing Note: Ensure each interaction ends with an offer for further assistance, such as: "Is there anything else I can assist you with?".

"""


     def initialize_conversation(prompt):
         if 'messagess' not in st.session_state:
             st.session_state.messagess = []
             st.session_state.messagess.append({"role": "system", "content": System_Prompt})
             chat =  openai.ChatCompletion.create(model = "gpt-4o-mini", messages = st.session_state.messagess, temperature=0.5, max_tokens=1500, top_p=1, frequency_penalty=0, presence_penalty=0)
             response = chat.choices[0].message.content
             st.session_state.messagess.append({"role": "assistant", "content": response})

     initialize_conversation(System_Prompt)

     for messages in st.session_state.messagess :
          if messages['role'] == 'system' : continue 
          else :
            with st.chat_message(messages["role"]):
                 st.markdown(messages["content"])

     if user_message := st.chat_input("Say something"):
         with st.chat_message("user"):
              st.markdown(user_message)
         query_embedding = get_embedding(user_message, engine='text-embedding-3-small')
         query_embedding_np = np.array([query_embedding]).astype('float32')
         _, indices = index.search(query_embedding_np, 2)
         retrieved_docs = [documents[i] for i in indices[0]]
         context = ' '.join(retrieved_docs)
         structured_prompt = f"Context:\n{context}\n\nQuery:\n{user_message}\n\nResponse:"
         chat =  openai.ChatCompletion.create(model = "gpt-4o-mini", messages = st.session_state.messagess + [{"role": "user", "content": structured_prompt}], temperature=0.5, max_tokens=1500, top_p=1, frequency_penalty=0, presence_penalty=0)
         st.session_state.messagess.append({"role": "user", "content": user_message})
         response = chat.choices[0].message.content
         with st.chat_message("assistant"):
              st.markdown(response)
         st.session_state.messagess.append({"role": "assistant", "content": response})