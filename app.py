import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain_community.llms import HuggingFaceHub
#embeddings
from langchain_community.embeddings import LocalAIEmbeddings

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings

#MI
import streamlit_dsfr as stdsfr

# CSS font family override
from streamlit_dsfr import override_font_family

import os

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"],
#, "."],
# " ", ""],
        chunk_size=400,
        chunk_overlap=80,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

from langchain_community.embeddings import HuggingFaceEmbeddings
def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="OrdalieTech/Solon-embeddings-large-0.1")
    #embeddings = HuggingFaceEmbeddings(model_name="mixedbread-ai/mxbai-embed-large-v1")

    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm_model = os.environ["OPENAI_API_MODEL"]="ai-chat"
#RAG-FR"
#vicuna-13b-16k"
#qwen2"
#RAG"
#llama3"
#phi3"
#wizardlm2"
    #os.environ["OPENAI_API_BASE"]="https://api-ai.ai-dev.fake-domain.name/v1"

    llm = ChatOpenAI(temperature=0.2,model_name=llm_model)

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
#search_kwargs={"k": 5}),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    print("conversation is:", st.session_state.conversation)

    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="Discutez avec plusieurs PDF",
                       page_icon=":books:")
    override_font_family()
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

#    st.header("Discutez avec plusieurs PDF :books:")
    user_question = stdsfr.dsfr_text_input("Posez une question sur vos documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Vos documents")
        pdf_docs = st.file_uploader(
            "Téléchargez vos PDF ici et cliquez sur 'Traiter'", accept_multiple_files=True)

        if st.button("Traiter"):
            with st.spinner("Traitement"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)


if __name__ == '__main__':
    main()
