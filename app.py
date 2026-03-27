import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.embeddings import Embeddings
from google import genai
from google.genai import types
from typing import List

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(page_title="RAG System", page_icon="🤖", layout="wide")

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;500;600&display=swap');
    .stApp { background: #0a0a0f; color: #e2e8f0; }
    h1, h2, h3 { font-family: 'Space Mono', monospace !important; color: #7dd3fc !important; }
    .main-header {
        font-family: 'Space Mono', monospace;
        font-size: 2.5rem; font-weight: 700;
        background: linear-gradient(135deg, #7dd3fc, #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header { color: #94a3b8; font-size: 1rem; margin-bottom: 2rem; }
    .chat-user {
        background: #1e293b; border-left: 3px solid #7dd3fc;
        padding: 12px 16px; border-radius: 8px; margin: 8px 0;
    }
    .chat-bot {
        background: #0f172a; border-left: 3px solid #a78bfa;
        padding: 12px 16px; border-radius: 8px; margin: 8px 0;
    }
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6, #8b5cf6);
        color: white; border: none; border-radius: 8px;
        font-family: 'Space Mono', monospace; font-weight: 700; width: 100%;
    }
    section[data-testid="stSidebar"] {
        background: #0f172a; border-right: 1px solid #1e293b;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  CUSTOM EMBEDDINGS using new google.genai
# ─────────────────────────────────────────────
class GeminiEmbeddings(Embeddings):
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            result = self.client.models.embed_content(
                model="text-embedding-004",
                contents=text
            )
            embeddings.append(result.embeddings[0].values)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        result = self.client.models.embed_content(
            model="text-embedding-004",
            contents=text
        )
        return result.embeddings[0].values


# ─────────────────────────────────────────────
#  FUNCTIONS
# ─────────────────────────────────────────────

def extract_text_from_pdfs(pdf_files):
    all_text = ""
    for pdf in pdf_files:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                all_text += text + "\n"
    return all_text


def split_into_chunks(raw_text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_text(raw_text)


def create_vector_store(chunks, api_key):
    embeddings = GeminiEmbeddings(api_key="AIzaSyBAWb7ewDiTKeFQpqEOkxc9gX9IXhyrPY4")
    return FAISS.from_texts(texts=chunks, embedding=embeddings)


def get_answer(question, vector_store, chat_history, api_key):
    # Get relevant docs
    docs = vector_store.similarity_search(question, k=4)
    context = "\n\n".join([doc.page_content for doc in docs])

    # Build chat history text
    history_text = ""
    for msg in chat_history[-6:]:
        if isinstance(msg, HumanMessage):
            history_text += f"Human: {msg.content}\n"
        else:
            history_text += f"AI: {msg.content}\n"

    prompt = f"""You are a helpful assistant. Answer the question based only on the context below.
If the answer is not in the context, say "I could not find that in the document."

Chat History:
{history_text}

Context from document:
{context}

Question: {question}

Answer:"""

    # Use new google.genai client
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-1.5-flash",
        contents=prompt
    )
    return response.text


# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "api_key" not in st.session_state:
    st.session_state.api_key = ""


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown("---")

    api_key = st.text_input(
        "🔑 Google Gemini API Key",
        type="password",
        placeholder="AIzaSy..."
    )

    st.markdown("---")
    st.markdown("### 📂 Upload Documents")

    pdf_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("🚀 Process Documents"):
        if not api_key:
            st.error("❌ Please enter your Gemini API Key")
        elif not pdf_files:
            st.error("❌ Please upload at least one PDF")
        else:
            with st.spinner("Processing your documents..."):
                try:
                    raw_text = extract_text_from_pdfs(pdf_files)
                    chunks = split_into_chunks(raw_text)
                    st.session_state.vector_store = create_vector_store(chunks, api_key)
                    st.session_state.api_key = api_key
                    st.session_state.pdf_processed = True
                    st.success(f"✅ Processed {len(pdf_files)} PDF(s) — {len(chunks)} chunks!")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

    st.markdown("---")

    if st.session_state.pdf_processed:
        st.markdown("✅ Documents loaded — Ready to chat!")

    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.session_state.vector_store = None
        st.session_state.pdf_processed = False
        st.rerun()

    st.markdown("---")
    st.markdown("""
    ### 📖 How to use:
    1. Enter Gemini API key
    2. Upload PDF files
    3. Click **Process Documents**
    4. Ask questions!

    ### 🆓 Get Free API Key:
    👉 aistudio.google.com/apikey
    """)


# ─────────────────────────────────────────────
#  MAIN AREA
# ─────────────────────────────────────────────
st.markdown('<p class="main-header">🤖 RAG System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Ask questions from your PDF documents using Gemini AI — Free!</p>',
            unsafe_allow_html=True)
st.markdown("---")

# Chat history
if st.session_state.chat_history:
    st.markdown("### 💬 Conversation")
    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            st.markdown(
                f'<div class="chat-user"><b>🧑 You:</b><br>{message.content}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="chat-bot"><b>🤖 AI:</b><br>{message.content}</div>',
                unsafe_allow_html=True
            )

# Question input
st.markdown("### 💭 Ask a Question")

if not st.session_state.pdf_processed:
    st.info("👈 Upload PDFs and click **Process Documents** in the sidebar first.")
else:
    question = st.text_input(
        "Type your question...",
        placeholder="What is the main topic of this document?"
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        ask_btn = st.button("Ask →")

    if ask_btn and question:
        with st.spinner("Thinking..."):
            try:
                answer = get_answer(
                    question,
                    st.session_state.vector_store,
                    st.session_state.chat_history,
                    st.session_state.api_key
                )
                st.session_state.chat_history.append(HumanMessage(content=question))
                st.session_state.chat_history.append(AIMessage(content=answer))
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
        st.rerun()