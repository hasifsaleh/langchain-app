import streamlit as st
import os

# from dotenv import load_dotenv
import toml
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template

# from langchain import HuggingFaceHub


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_text(raw_text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings(open_api_key=st.secrets("OPENAI_API_KEY"))
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(
    #     repo_id="google/flan-t5-xxl",
    #     model_kwargs={"temperature": 0.5, "max_length": 512},
    # )
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(), memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
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
                bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True
            )


# Function to check user credentials
def authenticate(username, password):
    # dev
    # valid_username = os.environ.get("ST_USERNAME")
    # valid_password = os.environ.get("ST_PASSWORD")
    # deploy
    valid_username = st.secrets["ST_USERNAME"]
    valid_password = st.secrets["ST_PASSWORD"]

    # Check if valid_username and valid_password are not None before comparison
    if valid_username is not None and valid_password is not None:
        return username == valid_username and password == valid_password
    else:
        return False


# Function to display the login page
def login():
    st.title("Login Page")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    # if username or password:
    if st.button("Login"):
        if authenticate(username, password):
            st.success("Login successful!")
            # Store login state in session state
            st.session_state.logged_in = True
            return st.session_state.logged_in
        else:
            st.error("Invalid username or password. Please try again.")


# Function to display the main content
def main_content():
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Document Detective :male-detective:")
    user_question = st.text_input("Ask a question about your documents")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True
        )
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)
                st.success("Processing Completed")

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)


# Main app logic
def main():
    # load_dotenv()
    st.set_page_config(page_title="Document Detective", page_icon=":mag:")

    # Check if logged in
    if not st.session_state.get("logged_in"):
        login()
    else:
        # Display main content if logged in is true
        main_content()


if __name__ == "__main__":
    main()
