import streamlit as st
import os
import time
from openai import OpenAI
from tavily import TavilyClient
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- 1. SETUP ---
st.set_page_config(page_title="HYDRO.AI", page_icon="🌊", layout="wide")

# Connect to NVIDIA and Tavily securely
client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=st.secrets["NVIDIA_API_KEY"])
tavily = TavilyClient(api_key=st.secrets["TAVILY_API_KEY"])

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
</style>
""", unsafe_allow_html=True)

st.title("🌊 HYDRO.AI")
st.caption("Next-gen intelligence for Water Resources Engineering.")

# --- 2. LOCAL VAULT (PDFs) ---
FAISS_INDEX_PATH = "wre_faiss_index"
DOCS_DIR = "wre_docs"

@st.cache_resource
def load_knowledge_base():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    if os.path.exists(FAISS_INDEX_PATH):
        return FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True), "Vault Active"
    
    if not os.path.exists(DOCS_DIR): os.makedirs(DOCS_DIR)
    loader = PyPDFDirectoryLoader(DOCS_DIR)
    docs = loader.load()
    if not docs: return None, "Awaiting PDFs..."
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=250)
    chunks = text_splitter.split_documents(docs)
    for chunk in chunks:
        chunk.page_content = f"SOURCE: {os.path.basename(chunk.metadata.get('source',''))}\n{chunk.page_content}"
        
    vector_db = FAISS.from_documents(chunks, embeddings)
    vector_db.save_local(FAISS_INDEX_PATH)
    return vector_db, f"Indexed {len(docs)} pages."

# --- 3. SIDEBAR ---
with st.sidebar:
    st.markdown("<h1 style='text-align: center;'>🌊</h1>", unsafe_allow_html=True)
    vector_db, status = load_knowledge_base()
    st.success(status)
    st.info("Connected to NVIDIA Gemma 4 & Tavily Web Search")
    
    st.divider()
    if st.button("🗑️ Reset Chat"):
        st.session_state.messages = []
        st.rerun()

# --- 4. AI BRAIN & CHAT UI ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Welcome! I am connected to your PDF Vault and the Live Internet. Ask me anything."}]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask HYDRO.AI...")

def stream_nvidia_response(system_prompt, user_query):
    try:
        # Ask NVIDIA Gemma 4 to stream the answer back smoothly
        response = client.chat.completions.create(
            model="google/gemma-4-31b-it",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            temperature=0.1,
            stream=True
        )
        for chunk in response:
            if chunk.choices and len(chunk.choices) > 0:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
                    
    except Exception as e:
        # If NVIDIA's server drops the connection mid-sentence, catch it gracefully!
        yield "\n\n*[Network connection interrupted by the NVIDIA server. The response above may be incomplete.]*"

if user_input:
    # 1. Show user message
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # 2. Gather Data (PDFs + Internet)
    with st.chat_message("assistant"):
        vault_data = ""
        web_data = ""
        
        with st.spinner("Searching Vault & Web..."):
            # Search Vault
            if vector_db:
                docs = vector_db.similarity_search(user_input, k=5)
                vault_data = "\n\n".join([d.page_content for d in docs])
            
            # Search Web
            try:
                search_results = tavily.search(query=user_input)
                web_data = "\n".join([f"- {r['content']}" for r in search_results['results']])
            except:
                web_data = "Web search currently unavailable."

        # 3. Create the Master Prompt
        system_prompt = (
            "You are a mentor for Water Resources Engineering. "
            "Answer the user using the provided VAULT DATA and WEB DATA below. "
            "Use LaTeX for math. Be professional and accurate.\n\n"
            f"=== VAULT DATA ===\n{vault_data}\n\n"
            f"=== WEB DATA ===\n{web_data}"
        )

        # 4. Stream the answer from NVIDIA
        full_response = st.write_stream(stream_nvidia_response(system_prompt, user_input))
        st.session_state.messages.append({"role": "assistant", "content": full_response})
