from langchain.llms.bedrock import Bedrock
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import boto3
import streamlit as st


#Bedrock client

bedrock_client =boto3.client(
    service_name= "bedrock-runtime",
    region_name= "us-east-1",
)

model_id = "ai21.j2-mid-v1"


llm=Bedrock(
    model_id= model_id,
    client= bedrock_client,
    model_kwargs= {"temperature": 0.9}  
)


def eesha_chatbot(language, user_text):
    prompt = PromptTemplate(
        input_variables=["language","user_text"],
        template="you are a chatbot.You are in {language}.\n\n{user_text}"
    )
    
    bedrock_chain = LLMChain(llm=llm, prompt=prompt)
    response=bedrock_chain({'language': language, 'user_text':user_text})
    
    return response




st.title("Bedrock Chatbot Demo")

language = st.sidebar.selectbox("Language",["english","spanish","hindi"])

if language:
    user_text = st.sidebar.text_area(label="what is your question?, max_char=100")
    
    
    
if user_text:
    response = eesha_chatbot(language, user_text)
    st.write(response['text'])