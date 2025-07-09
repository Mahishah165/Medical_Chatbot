import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Load FAISS vector DB
DB_FAISS_PATH = "vectorstore/db_faiss"
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# 2. Load the model and tokenizer
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

# 3. Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️ Using device: {device}")
model.to(device)

# 4. Ask user
user_query = input("Ask your question: ")

# 5. Retrieve top-k docs
retriever = db.as_retriever(search_kwargs={"k": 2})  # reduce to 2 for faster prompt
retrieved_docs = retriever.invoke(user_query)
context = "\n\n".join([doc.page_content[:500] for doc in retrieved_docs])  # trim long docs

# 6. Build prompt
prompt = f"""<|system|>
You are a compassionate and knowledgeable mental health assistant helping someone who is dealing with alcohol addiction.
Only use the given context.

<|user|>
Context:
{context}

Question: {user_query}

<|assistant|>
"""

# 7. Tokenize and generate
inputs = tokenizer(prompt, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():  # disable gradients for faster inference
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,  # reduced for faster output
        temperature=0.7,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

# 8. Decode
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
answer = response.split("<|assistant|>")[-1].strip()

print("\n🧠 RESPONSE:")
print(answer)

#print("\n📚 SOURCES:")
#for doc in retrieved_docs:
 #   print(f"- {doc.metadata.get('source', 'Unknown')}")
