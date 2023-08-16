import streamlit as st
import pandas as pd
import openai
from langchain.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI
from dotenv import load_dotenv, find_dotenv
import os
_ = load_dotenv(find_dotenv()) # read local .env file
if not os.environ.get('OPENAI_API_KEY'):
    st.error("Please define a .env file and insert OPENAI_API_KEY to it")
    st.stop()
openai.api_key = os.environ['OPENAI_API_KEY']
st.title('Your Personal LLM Data Analyst')

uploaded_file = st.file_uploader("Drop your CSV file here")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Here's a pick of your DataFrame.")
    st.write(df.head(2))

    st.write("Ask me what you want to know")
    question = st.text_input("question")
    if question:
        pd_agent = create_pandas_dataframe_agent(OpenAI(temperature=0), 
                                                df, 
                                                verbose=True)
        with st.spinner('Analysing ...'):
            response = pd_agent.run(question)
            st.write(response)
