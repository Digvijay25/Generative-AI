from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser


load_dotenv()

llm = ChatOpenAI(model='gpt-4')

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a facts expert who knows facts about {animal}."),
         ("human", 'Tell me {fact_count} facts')
    ]
)

chain = prompt_template | llm | StrOutputParser()

result = chain.invoke({'animal': 'cat', 'fact_count': 2})

print(result)