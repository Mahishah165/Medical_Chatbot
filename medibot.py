import os
import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Constants
DB_FAISS_PATH = "vectorstore/db_faiss"
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Cache the vector store
@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

# Cache the tokenizer, but move model to CUDA dynamically
@st.cache_resource
def load_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_ID)

def load_model():
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    return model, device

def build_prompt(context, question):
    return f"""<|system|>
You are a compassionate and knowledgeable mental health assistant helping someone who is dealing with alcohol addiction.
Answer clearly, kindly, and only using the context provided.

<|user|>
Context:
{context}

Question: {question}

<|assistant|>
"""

def generate_response(tokenizer, model, device, prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,  # reduced for speed
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("<|assistant|>")[-1].strip()

# Streamlit UI
def main():
    st.title("🧠 Medical Chatbot (Offline - Local LLM + RAG)")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    user_prompt = st.chat_input("Ask your medical question...")

    if user_prompt:
        st.chat_message("user").markdown(user_prompt)
        st.session_state.messages.append({"role": "user", "content": user_prompt})

        try:
            with st.spinner("💬 Thinking..."):
                vectorstore = get_vectorstore()
                tokenizer = load_tokenizer()
                model, device = load_model()

                # Retrieve context
                retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
                docs = retriever.invoke(user_prompt)

                # Trim context to 400 chars each
                context = "\n\n".join([doc.page_content[:400] for doc in docs])
                prompt = build_prompt(context, user_prompt)

                result = generate_response(tokenizer, model, device, prompt)
                sources = "\n".join([f"- {doc.metadata.get('source', 'Unknown')}" for doc in docs])

                final_output = result + "\n\n📚 **Sources:**\n" + sources
                st.chat_message("assistant").markdown(final_output)
                st.session_state.messages.append({"role": "assistant", "content": final_output})

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
