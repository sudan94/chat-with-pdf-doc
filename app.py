import streamlit as st
import os
from typing import Annotated, TypedDict
from PyPDF2 import PdfReader
import docx
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from streamlit_chat import message



# "with" notation
def main():
    load_dotenv()
    st.set_page_config(page_title="Chat With any files")
    st.header("💬 Chatbot")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    with st.sidebar:
        uploaded_files =  st.file_uploader("Upload your file",type=['pdf','docx'],accept_multiple_files=True)
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        process = st.button("Process")
    if process:
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()
        files_text = get_files_text(uploaded_files)
        # get text chunks
        text_chunks = get_text_chunks(files_text)
        # create vetore stores
        vetorestore = get_vectorstore(text_chunks)
         # create conversation chain
        st.session_state.conversation = get_conversation_chain(vetorestore,openai_api_key) #for openAI
        # st.session_state.conversation = get_conversation_chain(vetorestore) #for huggingface

        st.session_state.processComplete = True

    if  st.session_state.processComplete == True:
        user_question = st.chat_input("Ask Question about your files.")
        if user_question:
            handel_userinput(user_question)





def get_files_text(uploaded_files):
    text = ""
    for uploaded_file in uploaded_files:
        split_tup = os.path.splitext(uploaded_file.name)
        file_extension = split_tup[1]
        if file_extension == ".pdf":
            text += get_pdf_text(uploaded_file)
        elif file_extension == ".docx":
            text += get_docx_text(uploaded_file)
        else:
            text += get_csv_text(uploaded_file)
    return text


def get_pdf_text(pdf):
    pdf_reader = PdfReader(pdf)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_docx_text(file):
    doc = docx.Document(file)
    allText = []
    for docpara in doc.paragraphs:
        allText.append(docpara.text)
    text = ' '.join(allText)
    return text

def get_csv_text(file):
    return "a"

def get_text_chunks(text):
    # spilit ito chuncks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=900,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    knowledge_base = FAISS.from_texts(text_chunks,embeddings)
    return knowledge_base

class ChatState(TypedDict):
    # add_messages appends new messages instead of overwriting the list
    messages: Annotated[list, add_messages]
    context: str


def get_conversation_chain(vetorestore,openai_api_key):
    llm = ChatOpenAI(api_key=openai_api_key, model = 'gpt-3.5-turbo',temperature=0)
    retriever = vetorestore.as_retriever()

    def retrieve(state: ChatState):
        # last message is the question the user just asked
        question = state["messages"][-1].content
        docs = retriever.invoke(question)
        return {"context": "\n\n".join(doc.page_content for doc in docs)}

    def generate(state: ChatState):
        system = SystemMessage(content=(
            "You are a helpful assistant. Answer the question using the context below. "
            "If the answer is not in the context, say you don't know.\n\n"
            f"Context:\n{state['context']}"
        ))
        answer = llm.invoke([system] + state["messages"])
        return {"messages": [answer]}

    # build the graph: START -> retrieve -> generate -> END
    graph = StateGraph(ChatState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    # MemorySaver keeps the chat history per thread_id (replaces ConversationBufferMemory)
    return graph.compile(checkpointer=MemorySaver())

def handel_userinput(user_question):
    # thread_id tells LangGraph which conversation to load from the checkpointer
    config = {"configurable": {"thread_id": "streamlit-session"}}
    response = st.session_state.conversation.invoke(
        {"messages": [HumanMessage(content=user_question)]},
        config=config,
    )
    st.session_state.chat_history = response['messages']
    # token counts come straight off the answer message
    usage = getattr(response['messages'][-1], "usage_metadata", None) or {}

    # Layout of input/response containers
    response_container = st.container()

    with response_container:
        for i, messages in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                message(messages.content, is_user=True, key=str(i))
            else:
                message(messages.content, key=str(i))
        st.write(f"Total Tokens: {usage.get('total_tokens', 0)}" f", Prompt Tokens: {usage.get('input_tokens', 0)}" f", Completion Tokens: {usage.get('output_tokens', 0)}")

                 # for i, message in enumerate(st.session_state.chat_history):
    #     if i % 2 == 0:
    #         st.write(user_template.replace("{{MSG}}",message.content),unsafe_allow_html=True)
    #     else:
    #         st.write(bot_template.replace("{{MSG}}",message.content),unsafe_allow_html=True)





if __name__ == '__main__':
    main()






