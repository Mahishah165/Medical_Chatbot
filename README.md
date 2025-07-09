
# 🧠 Medical Chatbot – Offline Q&A with RAG and TinyLlama

This is an **offline medical chatbot** that answers health-related questions based on context retrieved from your local medical document dataset.

It uses:
- 🔍 FAISS for semantic document search
- 🧠 TinyLlama model loaded via `transformers` for local inference
- 💬 Streamlit for an interactive chat interface

---

## 📂 Project Structure

```
medical-chatbot/
├── main.py                    # Streamlit chat interface
├── vectorstore/
│   └── db_faiss/              # FAISS index built from medical documents
├── models/
│   └── TinyLlama-1.1B/        # Local model directory from Hugging Face
├── requirements.txt
└── .gitignore
```

---

## 🚀 How to Run Locally

### 1. Clone the Repo

```bash
git clone https://github.com/your-username/medical-chatbot.git
cd medical-chatbot
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the Model

Download the [TinyLlama-1.1B-Chat model](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0) and place it under the `models/` folder like this:

```
models/TinyLlama-1.1B-Chat-v1.0/
```

> You can use `from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")` if you're online, or download manually for offline use.

---

## ✅ Run the App

```bash
streamlit run main.py
```

A browser window will open with the chatbot interface. You can type questions like:

- *How do I reduce alcohol cravings?*
- *What are common symptoms of fever?*

The app will retrieve relevant info from your documents and generate an answer using the model.

---

## 📚 Features

- Works entirely offline
- Uses a retrieval-augmented generation (RAG) approach
- Responds based only on your dataset
- Designed for medical Q&A on topics like alcohol addiction, cough, fever, etc.

---

## 🛠 Tech Stack

| Component      | Tool                         |
|----------------|------------------------------|
| Model          | TinyLlama 1.1B Chat          |
| Inference      | Hugging Face Transformers    |
| Vector Search  | FAISS                        |
| Embeddings     | `all-MiniLM-L6-v2`           |
| UI             | Streamlit                    |

---

## 🙋‍♀️ Notes

- Ensure the FAISS vectorstore is pre-built using your medical dataset.
- The app may take some time (1–2 min) for the model to respond, depending on your system specs.
