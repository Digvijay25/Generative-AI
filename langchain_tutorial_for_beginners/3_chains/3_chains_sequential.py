from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableSequence

load_dotenv() 

model = ChatOpenAI(model='gpt-4')

animal_facts_template = ChatPromptTemplate.from_messages(
    [
        ('system', 'You like telling facts and you tell facts about {animals}'),
        ('human', 'Tell me {count} facts')
    ]
)

translation_template = ChatPromptTemplate.from_messages(
    [
        ('system', 'You are a translator and convert the provided text into {language}.'),
        ('human', 'Translate the following text to {language}:{text}')
    ]
)

prepare_for_translation = RunnableLambda(lambda output: {'text': output, 'language': 'french'})

chain = animal_facts_template | model | StrOutputParser() | prepare_for_translation | translation_template | model | StrOutputParser() 

result = chain.invoke({'animals': 'cat', 'count': 2})

print(result)