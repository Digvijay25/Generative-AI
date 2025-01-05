from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

# Create a new instance of the ChatOpenAI class
llm = ChatOpenAI(model="gpt-4o")

messages = [
    SystemMessage('You are an expert in social media content strategy'),
    HumanMessage('Give a short tip to create engaging posts on Instagram')
]


results = llm.invoke(messages)
print(results.content)