# import the packages
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores import FAISS
# from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI


OPENAI_API_KEY = "XXX-YYY-ZZZ"
TIKTOKEN_CACHE_DIR = "Folder_Path/GenAI/Library/cl100k_base.tiktoken"

# Define Heading to the page
st.header("Let's Roll !!!")

####### LLM Flow
# Upload the PDF document
with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload a PDF file and start asking questions", type='pdf')

# Read the PDF and extract the text
if file is not None:
    pdf_reader = PdfReader(file)
    text=""
    for page in pdf_reader.pages:
        text+=page.extract_text()
        # st.write(text)

# Break the text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    # st.write(chunks)

# Create Embeddings for each chunks
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Store the embeddings into Vector Store
    vector_store = FAISS.from_texts(chunks, embeddings)


####### User Flow
# Get User Question
    user_question = st.text_input("Type your question here")

# Do similarity search
    if user_question:
        match = vector_store.similarity_search(user_question)
        # st.write(match)

# Define the LLM
        llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY,
            temperature=0,
            max_tokens=1000,
            model_name="gpt-3.5-turbo"
        )

# Output results
# chain --> Take the question, get relevant document, pass it to the llm, generate the output
        chain = load_qa_chain(llm=llm, chain_type="stuff")
        response = chain.run(input_documents=match, question=user_question)
        st.write(response)

