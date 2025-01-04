from swarm import Agent, Swarm
import os
from typing import List
from pathlib import Path
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAI
import numpy as np
import openai
import faiss
from duckduckgo_search import DDGS
import pandas as pd

os.environ["OPENAI_API_KEY"] = " "

class PDFSwarmExtractor:
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.llm = OpenAI()  # Removed temperature argument
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200
        )

    def process_single_pdf(self, pdf_path: str) -> List[str]:
        try:
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()
            chunks = self.text_splitter.split_documents(pages)
            texts = [chunk.page_content for chunk in chunks]
            print(f"Successfully processed {pdf_path}")
            return texts
        except Exception as e:
            print(f"Error processing {pdf_path}: {str(e)}")
            return []

    def process_pdf_directory(self, directory_path: str) -> dict:
        pdf_files = [str(f) for f in Path(directory_path).glob("**/*.pdf")]
        results = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_pdf = {executor.submit(self.process_single_pdf, pdf): pdf for pdf in pdf_files}
            
            for future in future_to_pdf:
                pdf_path = future_to_pdf[future]
                try:
                    texts = future.result()
                    results[pdf_path] = texts
                except Exception as e:
                    print(f"Error processing {pdf_path}: {str(e)}")
                    results[pdf_path] = []
        
        return results

# Function to get embeddings
def get_embedding(text, model="text-embedding-ada-002"):
    response = openai.Embedding.create(input=[text], model=model)
    return response['data'][0]['embedding']

# Function to create embeddings and retrieve relevant documents
def create_embeddings_and_retrieve(query, texts):
    embeddings = [get_embedding(doc) for doc in texts]
    embedding_dim = len(embeddings[0])
    embeddings_np = np.array(embeddings).astype('float32')

    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings_np)

    query_embedding = get_embedding(query)
    query_embedding_np = np.array([query_embedding]).astype('float32')

    _, indices = index.search(query_embedding_np, 2)  # Retrieve top 2 documents
    retrieved_docs = [texts[i] for i in indices[0]]
    return ' '.join(retrieved_docs)

# Function to generate project ideas based on context
def generate_project_ideas(context):
    prompt = f"""
    Based on the following context, generate innovative project ideas:
    {context}
    
    Please provide a list of project ideas that are relevant and actionable.
    """

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.7,  # Set temperature here
        messages=[
            {"role": "system", "content": "You are a project idea generator."},
            {"role": "user", "content": prompt}
        ]
    )

    project_ideas = response.choices[0].message.content
    
    return project_ideas




def main():
    # Make sure you have set your OpenAI API key in .env file
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set your OPENAI_API_KEY in the .env file")
        return

    # Initialize the extractor
    extractor = PDFSwarmExtractor(max_workers=4)
    
    # Example usage
    pdf_directory = "/content/Open-source-LLMs-finetuning (1).pdf"  # Change this to your PDF directory
    if not os.path.exists(pdf_directory):
        os.makedirs(pdf_directory)
        print(f"Created directory: {pdf_directory}")
        print("Please place your PDF files in this directory")
        return
    
    # Process all PDFs in the directory
    results = extractor.process_pdf_directory(pdf_directory)
    
    # Initialize the Swarm client
    client = Swarm()

    
    def transfer_to_project_generator():
      return project_generator_agent

    # Initialize the Project Idea Generator Agent
    project_generator_agent = Agent(
        name="Project Idea Generator Agent",
        model="gpt-4o-mini",
        instructions="""
        Generate project ideas based on lessons learned from an AI Engineering Bootcamp. 
        Consider various aspects such as machine learning, data analysis, and AI ethics.
        """,
        functions=[
            create_embeddings_and_retrieve,  # Pass the function directly
            generate_project_ideas  # Pass the function directly
        ]
    )
    
    # Function to web search
    def web_search(query):
      results = DDGS().text(
          keywords = query,
          region = 'wt-wt',
          safesearch = 'off',
          timelimit = '7d',
          max_results = 10
      )

      results_df = pd.DataFrame(results)

      return results_df

    def transfer_to_python_expert():
      return python_expert_agent

    
    # Web search agent
    web_search_agent = Agent(
    name="Web Search Agent",
    instructions="You are a website search agent specialized in searching website content.",
    functions=[web_search, transfer_to_python_expert],
    )

    # Python Library Expert agent
    python_expert_agent = Agent(
        name="Python Expert Agent",
        model="gpt-4o-mini",
        instructions="""
        You are a Python Expert AI assistant. Your task is to suggest Python libraries that are highly suitable for the given query. Provide a brief explanation for each library you recommend, focusing on why it is relevant to the topic and highlighting key features.

        Guidelines:
        Only output suggestions for Python libraries. Do not mention tools, methods, or resources that are not directly related to Python.
        If no suitable Python library exists for the query, respond with:
        'There are no relevant Python libraries for this query.'
        Example Input Queries and Outputs:
        Query: 'Machine learning model training'
        Output:

        Scikit-learn: A user-friendly library for building and training machine learning models, including classification, regression, and clustering algorithms.
        TensorFlow: A powerful library for deep learning and advanced ML workflows, ideal for large-scale model training.
        Optuna: Useful for hyperparameter tuning during training to optimize performance.
        Query: 'Data visualization for time series'
        Output:

        Matplotlib: A versatile library for creating static, animated, and interactive plots.
        Seaborn: Built on Matplotlib, it provides an easier way to create aesthetically pleasing visualizations, particularly for statistical data.
        Plotly: Enables interactive and dynamic time series visualizations for dashboards or presentations.
        Query: 'Non-Python-related task'
        Output:
        'There are no relevant Python libraries for this query.'

        Use this format for your response. Be concise, clear, and only suggest Python libraries.
    """
    )

    def transfer_to_breakdown_agent():
      return code_breakdown_agent

    # code breakdown agent
    code_breakdown_agent = Agent(
    name = 'Code Breakdown Agent',
    instructions = 'You are a Code Breakdown Agent tasked with analyzing and explaining complex code to bootcamp participants in a clear, beginner-friendly manner. Your goal is to simplify code by breaking it into digestible parts, providing context, defining terms, and offering examples. All analysis must be in Markdown format for readability. Start with a high-level overview of the code’s purpose, then break it down step-by-step, explaining functionality, syntax, and intent. Avoid jargon unless explained, highlight key concepts or patterns, and use visuals like pseudocode or diagrams when helpful. Address potential confusion, suggest improvements or alternatives if applicable, and end with a concise summary of the code’s purpose and relevance.',
    model='gpt-4o-mini'
    )


    # orchestrator agent
    student_support = Agent(
    name = 'Student Support Agent',
    instructions = '''
    You are a student support agent that accepts bootcamp student's requests and calls a tool to transfer to the right intent.
    Once you are ready to transfer to the right intent, call the tool to transfer to the right intent.
    You dont need to know specifics, just the topic of the request.
    If the student request is about asking about project ideas, transfer to the Project Generator Agent.
    If the student request is about explaining code, transfer to the Code Breakdown Agent.
    If the student request is about asking python libraries and documentation, transfer to the Python Expert Agent.
    When you need more information to orchestrate the request to an agent, ask a direct question without explaining why you're asking it.
    Do not share your thought process with the user! Do not make unreasonable assumptions on behalf of user.
    ''',
    functions=[transfer_to_project_generator, transfer_to_breakdown_agent, transfer_to_python_expert]
)

    # Generate project ideas based on the extracted texts
    for pdf_path, texts in results.items():
        print(f"\nProcessed {pdf_path}:")
        print(f"Extracted {len(texts)} text chunks")
        if texts:
            context = create_embeddings_and_retrieve("machine learning", texts)  # Example query
            
            # Run the Project Idea Generator Agent
            project_ideas_response = client.run(
                agent=project_generator_agent,
                messages=[{
                    "role": "user",
                    "content": f"Generate project ideas based on the following context: {query_text}"
                }]
            )
            
            print("Generated Project Ideas:")
            for idea in project_ideas_response.messages[-1]["content"].strip().split('\n'):
                print("-", idea)

    query_text = "Generate project ideas for machine learning"
    
    struct = [{"role": "user", "content": query_text}]
    response = client.run(agent=student_support, messages=struct)
    struct.append({"role":"assistant", "content": response.messages[-1]["content"]})

    print(response.messages[0]["content"])

if __name__ == "__main__":
    main()