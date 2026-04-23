import streamlit as st
import os
import time
import base64
from openai import OpenAI
from tavily import TavilyClient
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- 1. CORE SETUP ---
st.set_page_config(page_title="HYDRO.AI", page_icon="🌊", layout="wide")

# Initialize NVIDIA & Tavily
client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=st.secrets["NVIDIA_API_KEY"])
tavily = TavilyClient(api_key=st.secrets["TAVILY_API_KEY"])

# Professional UI CSS
st.markdown("""
<style>
    footer {visibility: hidden;}
    [data-testid="stHeader"] { background-color: transparent !important; }
    .stApp h1 {
        background: -webkit-linear-gradient(45deg, #00C9FF, #005C97);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900;
        font-size: 3.5rem;
    }
    .stChatInputContainer { border-radius: 25px !important; }
</style>
""", unsafe_allow_html=True)

st.title("🌊 HYDRO.AI")
st.caption("Next-gen intelligence for Water Resources Engineering. | LDCE Mentor Mode")

# --- 2. THE KNOWLEDGE VAULT (RAG) ---
FAISS_INDEX_PATH = "wre_faiss_index"
DOCS_DIR = "wre_docs"

@st.cache_resource
def load_knowledge_base():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    if os.path.exists(FAISS_INDEX_PATH):
        return FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True), "Memory Active"
    
    if not os.path.exists(DOCS_DIR): os.makedirs(DOCS_DIR)
    loader = PyPDFDirectoryLoader(DOCS_DIR)
    docs = loader.load()
    if not docs: return None, "Awaiting Data..."
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=250)
    chunks = text_splitter.split_documents(docs)
    
    # Bug Fix: Stamp filename on every chunk
    for chunk in chunks:
        chunk.page_content = f"SOURCE: {os.path.basename(chunk.metadata.get('source',''))}\n{chunk.page_content}"
        
    vector_db = FAISS.from_documents(chunks, embeddings)
    vector_db.save_local(FAISS_INDEX_PATH)
    return vector_db, f"Indexed {len(docs)} pages."

# --- 3. THE MULTIMODAL MENTOR BRAIN ---
def process_query(query, image_file=None):
    context = ""
    search_results = ""
    
    # A. Search the Vault (PDFs)
    if vector_db:
        docs = vector_db.similarity_search(query, k=10)
        context = "\n\n".join([d.page_content for d in docs])
    
    # B. Search the Internet (Tavily)
    with st.spinner("Searching the internet for the latest data..."):
        web_data = tavily.search(query=query, search_depth="advanced")
        search_results = "\n".join([f"{r['title']}: {r['content']}" for r in web_data['results']])

    # C. Prepare the Vision Prompt (if image uploaded)
    messages = [
        {"role": "system", "content": (
            "You are a mentor of the WRE Dept at L.D. College of Engineering. "
            "Use LaTeX for math ($...$). Use Markdown tables for data. "
            "Synthesize information from the Vault and the Internet. "
            f"VAULT DATA:\n{context}\n\nINTERNET DATA:\n{search_results}"
        )},
    ]
    
    if image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": query},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        })
    else:
        messages.append({"role": "user", "content": query})

    # D. Call NVIDIA Gemma 4
    response = client.chat.completions.create(
        model="google/gemma-4-31b-it",
        messages=messages,
        temperature=0.1,
        stream=True
    )
    return response

# --- 4. SIDEBAR & UI ---
with st.sidebar:
    st.markdown("<h1 style='text-align: center;'>🌊</h1>", unsafe_allow_html=True)
    vector_db, status = load_knowledge_base()
    st.success(status)
    
    st.divider()
    st.subheader("🧠 Expand Brain")
    uploaded_pdf = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    if uploaded_pdf and st.button("Update Vault"):
        # (Same logic as before to rebuild index)
        st.cache_resource.clear()
        st.rerun()

    if st.button("🗑️ Reset Chat"):
        st.session_state.messages = []
        st.rerun()

# --- 5. CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Welcome! NVIDIA Gemma 4 is online with Internet Search and PDF Knowledge. How can I help your research today?"}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Vision Feature: Image uploader in the main chat
img_upload = st.file_uploader("📸 Analysis: Upload an image (Hydrograph, Map, etc.)", type=["jpg", "png", "jpeg"])

user_input = st.chat_input("Ask HYDRO.AI...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        response_stream = process_query(user_input, img_upload)
        full_response = st.write_stream(response_stream)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
