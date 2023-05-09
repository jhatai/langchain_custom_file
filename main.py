import streamlit as st
from langchain import PromptTemplate
from langchain.llms import OpenAI
import openai
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain import OpenAI, VectorDBQA
import os.path
import pathlib

st.header("Human query for custom file")
img_url = "https://media.discordapp.net/attachments/984632500875821066/1105248540336345258/HHT_A_robot_study_a_bible_hyper_realistic_dark_academia_theme_h_b5191b37-757b-452d-b5b7-5362c1cb588f.png?width=941&height=471"
st.image(image=img_url, use_column_width='auto')

import_result = []
result_status = "fail"
err_text = ""


def get_api_key():
    input_text = st.text_input(
        label="OpenAI API Key ",  placeholder="Ex: sk-2twmA8tfCb8un4...", key="openai_api_key_input")
    return input_text


openai_api_key = get_api_key()


# def query_similarity(similarity_text, faiss_index):
#     # query = "What did the president say about Ketanji Brown Jackson"
#     docs = faiss_index.similarity_search(similarity_text)
#     st.code(docs[0].page_content)


# def show_similarity_search_form(faiss_index):
#     with st.form(key='similarity_form'):
#         similarity_text = st.text_input(label='Enter similarity')
#         similarity_submit_button = st.form_submit_button(
#             label='Submit', on_click=query_similarity, args=["What is PySpark", faiss_index])


def query_human(faiss_index):
    # query = "What did the president say about Ketanji Brown Jackson"
    qa = VectorDBQA.from_chain_type(llm=OpenAI(
        openai_api_key=openai_api_key), chain_type="stuff", vectorstore=faiss_index)

    st.code(qa.run(st.session_state.human_question))


def get_human_text():
    input_text = st.text_input(label="Enter human question", label_visibility='collapsed',
                               placeholder="Ask a question", key="human_question")
    return input_text


# human_question= get_human_text()

# human_question = get_human_text()


def show_human_search_form(faiss_index):
    with st.form(key='human_form'):
        human_question = get_human_text()
        # st.code(f"human_question is {human_question}")
        st.form_submit_button(
            label='Submit', on_click=query_human, args=[faiss_index])
        # similarity_submit_button = st.form_submit_button(
        #     label='Submit', on_click=query_human, args=[human_question, faiss_index])


def import_file():
    faiss_index = None
    try:
        # loader = PyPDFLoader("docs/Data_Analysis_with_Python_and_PySpark.pdf")
        loader = PyPDFLoader("https://drive.google.com/file/d/1wTYmXZ-ZnmCnGsu9NK3U7VqZ5VV90dp1/view?usp=share_link")
        pages = loader.load_and_split()
        faiss_index = FAISS.from_documents(
            pages, OpenAIEmbeddings(openai_api_key=openai_api_key))
        page_num = str(len(pages))
        result_status = "sucess"
        err_text = f"import is done with {page_num} pages"
    except Exception as e:
        print(e)
        result_status = "fail"
        err_text = e
    st.write([result_status, err_text])
    if faiss_index:
        # show_similarity_search_form(faiss_index)
        show_human_search_form(faiss_index)

    # import_result= [result_status, faiss_index, text]


with st.form(key='import_form'):
    # url_input = st.text_input(label='Enter your name',key="file_url")
    import_submit_button = st.form_submit_button(
        label='Start import PDF file', on_click=import_file)

