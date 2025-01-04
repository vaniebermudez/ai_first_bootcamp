import os
import pandas as pd
from swarm import Agent, Swarm
from openai import OpenAI
from duckduckgo_search import DDGS


os.environ['OPENAI_API_KEY'] = ' '
api = OpenAI(api_key=" ")

client = Swarm(api)

def web_search(query):
  """
  Searching Web through Duckduckgo about the query then outputs a dataframe of the top 10 results.
  """
  search_text = 'Best Python Library for creating AI Agentic Framework'
  results = DDGS().text(
      keywords = search_text,
      region = 'wt-wt',
      safesearch = 'off',
      timelimit = '7d',
      max_results = 10
  )

  results_df = pd.DataFrame(results)

  return results_df

def handoff_to_python_expert_agent():
    """Hand off the search result to the python expert agent."""
    return python_expert_agent


web_search_agent = Agent(
    name="Web Search Agent",
    instructions="You are a website scraper agent specialized in scraping website content.",
    functions=[web_search, handoff_to_python_expert_agent],
)

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


query_text = 'What Python library to use for AI embeddings with PDF as input using the latest openAI version?'
struct = [{"role": "user", "content": query_text}]

response = client.run(agent=python_expert_agent, messages=struct)
struct.append({"role":"assistant", "content": response.messages[-1]["content"]})

print(response.messages[0]["content"])