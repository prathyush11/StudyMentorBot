import os
import streamlit as st
import asyncio
import tempfile
import time
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict

# LangChain and LangGraph Imports
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages

# Load API Key
try:
    load_dotenv()
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        st.error("GOOGLE_API_KEY not found. Please create a .env file and add your key.")
        st.stop()
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
except Exception as e:
    st.error(f"Error loading environment variables: {e}")
    st.stop()

# Rate Limit and Accuracy Helpers
def robust_api_invoke(api_func, *args, max_retries=5, backoff_base=2, **kwargs):
    for attempt in range(max_retries):
        try:
            return api_func(*args, **kwargs)
        except Exception as e:
            error_msg = str(e).lower()
            if "rate limit" in error_msg or "too many requests" in error_msg:
                wait_sec = backoff_base ** attempt
                st.warning(f"Rate limit hit. Retrying in {wait_sec} seconds...")
                time.sleep(wait_sec)
            else:
                st.error(f"Unexpected error: {e}")
                break
    st.error("Operation failed several times due to rate limits or errors. Please try again later.")
    return None

def robust_llm_invoke(messages, max_retries=5, backoff_base=2):
    response = robust_api_invoke(llm_with_tools.invoke, messages, max_retries=max_retries, backoff_base=backoff_base)
    if response is not None and hasattr(response, 'answer'):
        answer_str = str(response.answer).strip().lower()
        if "i don't know" in answer_str or not answer_str:
            st.warning("The answer may be incomplete or uncertain. Try asking in a different way.")
    return response

def robust_rag_invoke(prompt, max_retries=5, backoff_base=2):
    response = robust_api_invoke(st.session_state.rag_chain.invoke, {"input": prompt}, max_retries=max_retries, backoff_base=backoff_base)
    if response is not None and "answer" in response:
        answer_str = str(response["answer"]).strip().lower()
        if "i don't know" in answer_str or not answer_str:
            st.warning("Document search did not find a confident answer. Asking a follow-up may help.")
    return response

# State
class StudyState(TypedDict):
    messages: Annotated[list, add_messages]
    current_quiz: dict
    finished: bool

# Tools
@tool
def create_quiz_question(question: str, options: list[str], correct_option: str) -> dict:
    """Creates a quiz question and returns it as a dictionary."""
    return {"question": question, "options": options, "correct_option": correct_option}

@tool
def parse_notes(notes: str) -> list[str]:
    """Extracts key concepts from study notes."""
    return [line.strip() for line in notes.splitlines() if line.strip()]

# @tool
# def confirm_session(response: str) -> bool:
#     """Determines if the user wants to end the session."""
#     return response.strip().lower() in ("yes", "y", "quit", "exit", "bye")

# Initialize LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
tools = [create_quiz_question, parse_notes]
llm_with_tools = llm.bind_tools(tools)

# System Prompt and Welcome Message
STUDYMENTOR_SYSINT = (
    "system",
    "You are StudyMentorBot, an AI tutor helping students study. You assist with a single topic at a time (e.g., mathematics, history, science)."
    "Users can specify a topic, upload study notes, or ask for quizzes, explanations, or resources. "
    "Use 'create_quiz_question' to generate quizzes. Use 'parse_notes' to extract concepts from notes. "
    "Always verify topics and concepts against user input or notes. If unsure, ask clarifying questions. "
    "Respond in a clear, encouraging, and teacher-like tone. Stay on-topic (no off-topic discussion). "
    "After quizzes or explanations, ask if the user wants to continue or end the session."
)

WELCOME_MSG = "Hey there, Need help with your studies, I GOT YOU. Welcome to StudyMentorBot. I am here to help you today. What topic do you need help with?"

# Graph Nodes and Router
def agent_node(state: StudyState) -> dict:
    messages_with_system_prompt = [STUDYMENTOR_SYSINT] + state["messages"]
    response = robust_llm_invoke(messages_with_system_prompt)
    if response is None:
        return {"messages": [AIMessage(content="Sorry, an error occurred with the AI model. Please try again.")]}

    if not getattr(response, "tool_calls", None):
        return {"messages": [response]}

    tool_output_messages = []
    quiz_data_to_store = {}

    for tool_call in response.tool_calls:
        tool_name = tool_call["name"]
        tool_to_call = next(t for t in tools if t.name == tool_name)
        tool_output = tool_to_call.invoke(tool_call["args"])

        if tool_name == "create_quiz_question":
            quiz_data_to_store = tool_output
            formatted_question = f"{tool_output['question']}\n\n"
            for i, opt in enumerate(tool_output['options']):
                formatted_question += f"{chr(97 + i)}) {opt}\n"
            tool_output_messages.append(AIMessage(content=formatted_question))
        else:
            tool_output_messages.append(AIMessage(content=str(tool_output)))

    return {"messages": tool_output_messages, "current_quiz": quiz_data_to_store}

def check_answer_node(state: StudyState) -> dict:
    user_answer_msg = state["messages"][-1].content
    correct_answer = state["current_quiz"]["correct_option"]
    if user_answer_msg.strip().lower() in correct_answer.strip().lower():
        result_message = "That's correct! Excellent work! 👍"
    else:
        result_message = f"Not quite. The correct answer was '{correct_answer}'."
    return {"messages": [AIMessage(content=result_message)], "current_quiz": {}}

def quiz_router(state: StudyState) -> str:
    if state.get("current_quiz") and state["current_quiz"]:
        return "check_answer_node"
    return "agent_node"

# Build Graph
graph = StateGraph(StudyState)
graph.add_node("agent", agent_node)
graph.add_node("check_answer_node", check_answer_node)
graph.add_edge(START, "agent")
graph.add_edge("check_answer_node", "agent")
graph.add_conditional_edges("agent", lambda x: "agent", {"agent": "agent"})
runner = graph.compile()

# Streamlit Web App
st.title("🎓 StudyMentorBot")
st.caption("Your AI-powered study partner, ready to help!")

# Sidebar for Document Upload and RAG
with st.sidebar:
    st.header("Upload Your Study Notes")
    st.markdown("Upload a PDF document and I can answer questions based on its content.")
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
    if uploaded_file:
        if st.button("Process Document"):
            with st.spinner("Processing document... This may take a moment."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
                    tmpfile.write(uploaded_file.getvalue())
                    tmp_path = tmpfile.name
                try:
                    # Fix for the asyncio event loop error in Streamlit
                    try:
                        loop = asyncio.get_running_loop()
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                    loader = PyPDFLoader(tmp_path)
                    documents = loader.load()
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    splits = text_splitter.split_documents(documents)
                    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                    vectorstore = FAISS.from_documents(splits, embeddings)
                    retriever = vectorstore.as_retriever()
                    rag_system_prompt = (
                        "You are an assistant for question-answering tasks. Use the following pieces of retrieved context "
                        "to answer the question. If you don't know the answer, just say that you don't know. "
                        "Keep the answer concise and helpful."
                        "\n\n{context}"
                    )

                    prompt_template = ChatPromptTemplate.from_messages(
                        [("system", rag_system_prompt), ("human", "{input}")]
                    )

                    qa_chain = create_stuff_documents_chain(llm, prompt_template)
                    st.session_state.rag_chain = create_retrieval_chain(retriever, qa_chain)
                    st.success("Document processed! You can now ask questions about it.")
                finally:
                    os.remove(tmp_path)

# Main Chat Interface
if "state" not in st.session_state:
    st.session_state.state = {"messages": [AIMessage(content=WELCOME_MSG)], "current_quiz": {}}

for msg in st.session_state.state['messages']:
    if isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)
    elif isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)

if prompt := st.chat_input("Ask a question about your document or a general topic."):
    st.chat_message("user").write(prompt)
    if "rag_chain" in st.session_state and st.session_state.rag_chain:
        # RAG Path
        with st.spinner("Searching your document..."):
            st.session_state.state['messages'].append(HumanMessage(content=prompt))
            response = robust_rag_invoke(prompt)
            if response is not None:
                bot_response = AIMessage(content=response["answer"])
            else:
                bot_response = AIMessage(content="Sorry, document search failed due to repeated errors or rate limits.")
            st.session_state.state['messages'].append(bot_response)
            st.chat_message("assistant").write(bot_response.content)
    else:
        # Original Graph Agent Path
        current_state = st.session_state.state
        current_state['messages'].append(HumanMessage(content=prompt))
        next_node = quiz_router(current_state)
        if next_node == "check_answer_node":
            updated_state = check_answer_node(current_state)
        else:
            updated_state = agent_node(current_state)
        st.session_state.state['messages'].extend(updated_state['messages'])
        st.session_state.state['current_quiz'] = updated_state.get('current_quiz', {})
        st.chat_message("assistant").write(updated_state['messages'][-1].content)
