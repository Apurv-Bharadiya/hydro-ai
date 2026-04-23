import streamlit as st
import os
from openai import OpenAI # NVIDIA uses the OpenAI-compatible format
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- 1. CONFIGURATION ---
# We switch from Groq to NVIDIA
client = OpenAI(
  base_url="https://integrate.api.nvidia.com/v1",
  api_key=st.secrets["NVIDIA_API_KEY"]
)

# ... [Keep your existing UI and Knowledge Base code exactly the same] ...

# --- 4. THE NVIDIA GEMMA 4 MENTOR BRAIN ---
def get_ai_response(user_input, context, chat_history):
    system_prompt = (
        "You are a dedicated mentor of the Water Resources Engineering Department at L. D. College of Engineering, Ahmedabad. "
        "You guide postgraduate students with encouraging, practical, and highly technical explanations.\n\n"
        f"=== DOCUMENT VAULT (PRIORITY) ===\n{context}\n\n"
        f"=== CONVERSATION HISTORY ===\n{chat_history}"
    )
    
    # This is the call to NVIDIA's Gemma 4 31B
    completion = client.chat.completions.create(
      model="google/gemma-4-31b-it",
      messages=[
          {"role": "system", "content": system_prompt},
          {"role": "human", "content": user_input}
      ],
      temperature=0.1,
      top_p=0.7,
      max_tokens=4096,
      stream=True
    )
    return completion

# ... [In your chat UI section, call this function instead of rag_chain.invoke] ...
