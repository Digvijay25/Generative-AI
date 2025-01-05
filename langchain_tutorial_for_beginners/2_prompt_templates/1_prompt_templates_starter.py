from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

load_dotenv() 

llm = ChatOpenAI(model='gpt-4')

template = "Write a {tone} email to {company} expressing interest in the {position} position, mentioning \
            {skill} as a key strength. Keep it to 4 lines max"

prompt_template = ChatPromptTemplate.from_template(template)

prompt = prompt_template.invoke({
    "tone": "energetic",
    "company": "samsung", 
    "position": "AI Engineer",
    "skill": "AI"
})

result = llm.invoke(prompt)
print(result.content) 

