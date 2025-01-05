from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableBranch


load_dotenv() 

model = ChatOpenAI(model='gpt-4o')

positive_feedback_template = ChatPromptTemplate.from_messages(
    [
        ('system', 'You are a helpful assistant.'), 
        ('human', 'Generate a thank you note for this positive feedback: {feedback}.')
    ]
)

negative_feedback_template = ChatPromptTemplate.from_messages(
    [
        ('system', 'You are a helpful assistant.'),
        ('human', 'Genearate a response addressing this negative feedback: {feedback}.')
    ]
)

neutral_feedback_template = ChatPromptTemplate.from_messages(
    [
        ('system', 'You are a helpful assistant.'),
        ('human', 'Genearate a response addressing this neutral feedback: {feedback}.')
    ]
)

escalate_feedback_template = ChatPromptTemplate.from_messages(
    [
        ('system', 'You are a helpful assistant.'),
        ('human', 'Genearate a response to escalate this feedback to a human agent: {feedback}.')
    ]
)

classification_template = ChatPromptTemplate.from_messages(
    [
        ('system', 'You are a helpful assistant.'),
        ('human', 'Classify the sentiment of this feedback as positive, negative, neutral, or escalate: {feedback}.')
    ]
)

branches = RunnableBranch(
    (
        lambda x: 'positive' in x,
        positive_feedback_template | model | StrOutputParser()
    ), 

    (
        lambda x: 'negative' in x, 
        negative_feedback_template | model | StrOutputParser()
    ), 

    (
        lambda x: 'neutral' in x, 
        neutral_feedback_template | model | StrOutputParser()
    ),

    escalate_feedback_template | model | StrOutputParser()

)

classification_chain = classification_template | model | StrOutputParser() 

chain = classification_chain | branches

review = 'The product is terrible. It broke after just one use and the quality is very poor.'

result = chain.invoke({'feedback': review})

print(result)