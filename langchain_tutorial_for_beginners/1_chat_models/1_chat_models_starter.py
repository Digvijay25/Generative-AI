from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# Create a new instance of the ChatOpenAI class
llm = ChatOpenAI(model="gpt-4o")

results = llm.invoke('What is the square root of 49?')

print(results.content)