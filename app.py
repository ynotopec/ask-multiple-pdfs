import os
from typing import List

import streamlit as st
import streamlit_dsfr as stdsfr
from dotenv import load_dotenv
from htmlTemplates import css, bot_template, user_template
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from PyPDF2 import PdfReader
from streamlit.runtime.uploaded_file_manager import UploadedFile
from streamlit_dsfr import override_font_family

def get_pdf_text(pdf_docs: List[UploadedFile]) -> str:
    pages_content = []
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                pages_content.append(page_text)
    return "\n".join(pages_content)


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"],
        chunk_size=400,
        chunk_overlap=80,
        length_function=len,
    )
    return text_splitter.split_text(text)


def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="OrdalieTech/Solon-embeddings-large-0.1"
    )
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)


def get_conversation_chain(vectorstore):
    llm_model = os.environ.get("OPENAI_API_MODEL", "ai-chat")
    llm = ChatOpenAI(temperature=0.2, model_name=llm_model)

    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(), memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    if not st.session_state.conversation:
        st.warning("Veuillez d'abord traiter vos documents avant de poser une question.")
        return

    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response["chat_history"]

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(
                user_template.replace("{{MSG}}", message.content),
                unsafe_allow_html=True,
            )
        else:
            st.write(
                bot_template.replace("{{MSG}}", message.content),
                unsafe_allow_html=True,
            )


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
            if not pdf_docs:
                st.warning("Veuillez télécharger au moins un document PDF.")
            else:
                with st.spinner("Traitement"):
                    raw_text = get_pdf_text(pdf_docs)

                    if not raw_text.strip():
                        st.warning("Impossible d'extraire du texte des fichiers fournis.")
                        return

                    text_chunks = get_text_chunks(raw_text)
                    vectorstore = get_vectorstore(text_chunks)
                    st.session_state.conversation = get_conversation_chain(
                        vectorstore)


if __name__ == '__main__':
    main()
