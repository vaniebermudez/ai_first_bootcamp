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
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
import faiss
import streamlit as st
import warnings
from streamlit_option_menu import option_menu
from streamlit_extras.mention import mention

warnings.filterwarnings("ignore")

st.set_page_config(page_title="InvyTrack: Your Smart Partner for Real-Time Inventory Precision", page_icon="", layout="wide")

# First, define the initialize_rag function
def initialize_rag():
    # Load and process the inventory dataset
    loader = CSVLoader('https://raw.githubusercontent.com/vaniebermudez/ai_first_bootcamp/refs/heads/main/Day4/inventory_products_dataset.csv')
    documents = loader.load()
    
    # Split documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    splits = text_splitter.split_documents(documents)
    
    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(splits, embeddings)
    
    return vectorstore

# Then define initialize_conversation
def initialize_conversation(prompt):
    if 'message' not in st.session_state:
        st.session_state.message = []
        st.session_state.message.append({"role": "system", "content": System_Prompt})
        
    # Initialize RAG components if not already initialized
    if 'vectorstore' not in st.session_state:
        st.session_state.vectorstore = initialize_rag()
    
    # Initialize qa_chain if not already initialized
    if 'qa_chain' not in st.session_state:
        # Create ChatOpenAI instance
        llm = ChatOpenAI(model="gpt-4", temperature=0.5)
        
        # Create ConversationalRetrievalChain
        st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True,
        )

with st.sidebar:
    try:
        st.image('images/invytrack.png')
    except:
        st.title("InvyTrack")  # Fallback title if image is missing
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

   st.title("InvyTrack: Your Smart Partner for Real-Time Inventory Precision")
   st.write("Welcome to InvyTrack, your comprehensive, AI-powered solution for real-time inventory management. Designed with cutting-edge technology, InvyTrack ensures that your supply chain runs seamlessly by providing instant insights, proactive alerts, and efficient stock control. Whether you're managing multiple warehouses or tracking a diverse product range, InvyTrack empowers you with the tools to optimize operations, reduce costs, and maintain perfect inventory levels. Say goodbye to manual errors and outdated systems—experience the future of supply chain management with InvyTrack, where precision meets innovation.")
   
elif options == "About Us" :
     st.title("About Us")
     st.write("# Vanessa Althea Bermudez")
     try:
        st.image('images/vanie.png')
     except:
        st.write('Vanie Bermudez')
     st.write("## AI Enthusiast / Data Scientist")
     st.text("Connect with me via Linkedin : https://www.linkedin.com/in/vaniebermudez/")
     st.text("Github : https://github.com/vaniebermudez/")
     st.write("\n")


elif options == "Model" :
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
        
        # Get chat history for context
        chat_history = [(msg["content"], st.session_state.message[i+1]["content"]) 
                        for i, msg in enumerate(st.session_state.message[:-1]) 
                        if msg["role"] == "user"]
        
        # Get response using RAG
        result = st.session_state.qa_chain({"question": user_message, "chat_history": chat_history})
        response = result["answer"]
        
        with st.chat_message("assistant"):
            st.markdown(response)
            # Optionally display sources
            if "source_documents" in result:
                with st.expander("Sources"):
                    for doc in result["source_documents"]:
                        st.write(doc.page_content)
        
        st.session_state.message.append({"role": "assistant", "content": response})
