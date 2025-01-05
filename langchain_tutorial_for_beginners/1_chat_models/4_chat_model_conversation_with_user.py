from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()

llm = ChatOpenAI(model='gpt-4o')

chat_history = []

system_message = SystemMessage(content='You are a helpful AI assistant.')
chat_history.append(system_message)

while True: 
    query = input('You: ')
    if query == 'exit':
        break

    chat_history.append(HumanMessage(content=query))
    result = llm.invoke(chat_history)
    response = result.content
    chat_history.append(AIMessage(content=response))

    print(f'AI: {response}')


print('-----Message History -----')
print(chat_history)
