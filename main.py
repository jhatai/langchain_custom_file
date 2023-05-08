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


# st.write(bytes_data)

# import pdf
# Form+ filepicker + commit + text with status
# Forms can be declared using the 'with' syntax
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
        loader = PyPDFLoader("data/0222_Jerry_Hung.pdf")
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
    text_input = st.text_input(label='Enter your name')
    import_submit_button = st.form_submit_button(
        label='Submit', on_click=import_file)


# st.write("""
# # File Picker
# """)
# uploaded_file = st.file_uploader("Choose a PDF file")

# def upload():
#     if uploaded_file is None:
#         st.session_state["upload_state"] = "Upload a file first!"
#     else:
#         # data = uploaded_file.getvalue().decode('utf-8')
#         parent_path = pathlib.Path(__file__).parent.parent.resolve()
#         save_path = os.path.join(parent_path, "data")
#         complete_name = os.path.join(save_path, uploaded_file.name)+"/"+uploaded_file.name

#         with open(complete_name, "wb") as destination_file:
#             destination_file.write(uploaded_file.getbuffer())
#             destination_file.close()
#             st.session_state["upload_state"] = "Saved " + complete_name + " successfully!"
#             st.session_state["file_name"] = uploaded_file.name
#     # st.write(st.session_state["file_name"])
# st.button("Upload file to Sandbox", on_click=upload)
