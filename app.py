import streamlit as st
import os
import time
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- 1. CONFIGURATION & UI ---
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

st.set_page_config(page_title="HYDRO.AI", page_icon="🌊", layout="wide", initial_sidebar_state="expanded")

# BUG FIX: Removed the CSS that was hiding the Streamlit top menu!
st.markdown("""
<style>
    /* Hide ONLY the bottom footer, keep the top menu visible for Light/Dark mode */
    footer {visibility: hidden;}
    
    /* Make the top header transparent so it looks clean, but buttons remain clickable */
    [data-testid="stHeader"] {
        background-color: transparent !important;
    }
    
    /* HYDRO.AI Gradient Title */
    .stApp h1 {
        background: -webkit-linear-gradient(45deg, #00C9FF, #005C97);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 900;
        letter-spacing: -1.5px;
        font-size: 3.5rem;
    }
    
    /* Sleek Chat Input */
    .stChatInputContainer {
        border-radius: 25px !important;
        border: 1px solid rgba(150, 150, 150, 0.3) !important;
        box-shadow: 0 4px 15px 0 rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

st.title("🌊 HYDRO.AI")
st.caption("Next-gen intelligence for Water Resources Engineering.")

# --- 2. DYNAMIC KNOWLEDGE BASE ---
FAISS_INDEX_PATH = "wre_faiss_index"
DOCS_DIR = "wre_docs"

if not os.path.exists(DOCS_DIR):
    os.makedirs(DOCS_DIR)

@st.cache_resource
def load_knowledge_base():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    if os.path.exists(FAISS_INDEX_PATH):
        vector_db = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        return vector_db, "System Online (Memory Active)"
        
    loader = PyPDFDirectoryLoader(DOCS_DIR)
    docs = loader.load()
    if not docs: return None, "Awaiting Data Vault Uploads..."
    
    valid_docs = [d for d in docs if len(d.page_content.strip()) > 10]
    
    # Split the documents FIRST
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=250)
    chunks = text_splitter.split_documents(valid_docs)
    
    # BUG FIX: Inject the filename into EVERY SINGLE CHUNK after splitting
    for chunk in chunks:
        filename = os.path.basename(chunk.metadata.get('source', 'Unknown'))
        chunk.page_content = f"--- SOURCE DOCUMENT: {filename} ---\n\n" + chunk.page_content
        
    vector_db = FAISS.from_documents(chunks, embeddings)
    vector_db.save_local(FAISS_INDEX_PATH)
    return vector_db, f"Indexed {len(valid_docs)} pages."

# --- 3. SIDEBAR: CONTROL CENTER ---
with st.sidebar:
    st.markdown("<h1 style='text-align: center; font-size: 4rem;'>🌊</h1>", unsafe_allow_html=True)
    
    st.header("⚙️ Core Systems")
    vector_db, status_msg = load_knowledge_base()
    if vector_db:
        st.success(status_msg)
    else:
        st.warning(status_msg)
        
    st.divider()
    
    st.subheader("🧠 Expand Brain")
    uploaded_files = st.file_uploader("Upload IS Codes/Manuals (PDF)", type="pdf", accept_multiple_files=True)
    if uploaded_files:
        if st.button("Process New Documents"):
            with st.spinner("Assimilating new data..."):
                for uploaded_file in uploaded_files:
                    with open(os.path.join(DOCS_DIR, uploaded_file.name), "wb") as f:
                        f.write(uploaded_file.getbuffer())
                if os.path.exists(FAISS_INDEX_PATH):
                    import shutil
                    shutil.rmtree(FAISS_INDEX_PATH)
                st.cache_resource.clear()
                st.rerun()

    st.divider()
    
    st.subheader("📥 Export Data")
    if "messages" in st.session_state and len(st.session_state.messages) > 1:
        chat_export = "# HYDRO.AI - Lab Session Report\n\n"
        for msg in st.session_state.messages:
            role = "🧑‍🎓 Student" if msg["role"] == "user" else "🤖 HYDRO.AI"
            chat_export += f"### {role}\n{msg['content']}\n\n"
            if msg.get("sources"):
                chat_export += f"*(Sources: {', '.join(msg['sources'])})*\n\n"
            chat_export += "---\n\n"
        
        st.download_button(
            label="Download Session Report (.md)",
            data=chat_export,
            file_name="HYDRO_AI_Session.md",
            mime="text/markdown",
            use_container_width=True
        )
    
    if st.button("🗑️ Reset Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history = ""
        st.rerun()

# --- 4. THE CONVERSATIONAL MENTOR BRAIN ---
llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.1) 

system_prompt = (
    "You are a dedicated mentor of the Water Resources Engineering Department at L. D. College of Engineering. "
    "You guide postgraduate students with encouraging, practical, and highly technical explanations.\n\n"
    "CRITICAL INSTRUCTIONS:\n"
    "1. Extract factual answers ONLY from the '=== DOCUMENT VAULT ===' section.\n"
    "2. If the user mentions a specific author (e.g., 'Bansal'), prioritize the text marked with that SOURCE DOCUMENT.\n"
    "3. If you find the answer but it's from a different source than requested, provide the answer anyway and cite the source you used. NEVER say 'I couldn't find the author'.\n"
    "4. Use LaTeX for math ($...$). Use Markdown tables for classifications.\n\n"
    "=== DOCUMENT VAULT ===\n{context}\n\n"
    "=== CHAT HISTORY ===\n{chat_history}"
)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

if vector_db:
    combine_docs_chain = create_stuff_documents_chain(llm, prompt_template)
    rag_chain = create_retrieval_chain(vector_db.as_retriever(search_kwargs={"k": 15}), combine_docs_chain)
else:
    rag_chain = None

# --- 5. ADVANCED CHAT UI ---
if "messages" not in st.session_state or len(st.session_state.messages) == 0:
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome! I am your mentor from the Water Resources Engineering Department at L.D. College of Engineering. I have our entire curriculum loaded and ready. What concept can I help you master today?", "sources": []}
    ]
    st.session_state.chat_history = ""

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("🔍 View Verified Sources"):
                st.write(", ".join(message["sources"]))

# Suggested Prompts
if len(st.session_state.messages) == 1:
    cols = st.columns(3)
    if cols[0].button("💧 Explain Ideal Fluids (Bansal)"):
        st.session_state.suggested_input = "Based on Bansal, what is an Ideal Fluid?"
    if cols[1].button("🌊 What is a Hydraulic Jump?"):
        st.session_state.suggested_input = "What is a hydraulic jump and where does it occur?"
    if cols[2].button("💻 How to run a 1D model in HEC-RAS"):
        st.session_state.suggested_input = "What are the steps to run a 1D steady flow model in HEC-RAS?"

user_input = st.chat_input("Ask HYDRO.AI a question...")
if "suggested_input" in st.session_state:
    user_input = st.session_state.suggested_input
    del st.session_state.suggested_input

def stream_data(text):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.015)

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input, "sources": []})

    with st.chat_message("assistant"):
        if rag_chain:
            with st.spinner("Consulting the engineering vault..."):
                response = rag_chain.invoke({
                    "input": user_input, 
                    "chat_history": st.session_state.chat_history
                })
                ai_reply = response["answer"]
                sources = list(set([os.path.basename(doc.metadata.get('source', 'Unknown')) for doc in response['context']]))
            
            st.write_stream(stream_data(ai_reply))
            
            if sources:
                with st.expander("🔍 View Verified Sources"):
                    st.write(", ".join(sources))
            
            st.session_state.messages.append({"role": "assistant", "content": ai_reply, "sources": sources})
            st.session_state.chat_history += f"Student: {user_input}\nHYDRO.AI: {ai_reply}\n\n"