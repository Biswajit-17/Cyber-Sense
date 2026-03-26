# CyberSense — AI Legal Advisor

**CyberSense** is an AI-powered legal assistant designed to provide guidance on Indian Cyber Laws. It uses **Retrieval-Augmented Generation (RAG)** to answer user queries by retrieving relevant legal clauses from a curated knowledge base of PDF documents and synthesizing responses using the **Groq** API (Llama 3).

## 🚀 Features

*   **RAG Architecture**: Retrieves accurate legal context from local PDF documents to ground AI responses and minimize hallucinations.
*   **Semantic Search**: Uses **ChromaDB** and **SentenceTransformers** (`all-MiniLM-L6-v2`) to find the most relevant legal sections based on user queries.
*   **Interactive Chat UI**: Built with **Streamlit**, featuring persistent chat history and a professional legal advisor persona.
*   **Document Ingestion Engine**: Automatically processes, splits, and embeds PDF legal documents for the knowledge base.

## 🛠️ Tech Stack

*   **Frontend**: Streamlit
*   **LLM Inference**: Groq API (Llama 3.1)
*   **Vector Database**: ChromaDB (Persistent storage)
*   **Embeddings**: SentenceTransformers (`all-MiniLM-L6-v2`)
*   **Data Processing**: LangChain (Text Splitting), PyPDF2
*   **Language**: Python 3.x

## 📂 Project Structure

```bash
CyberSense/
├── app.py                # Main Streamlit application entry point
├── requirements.txt      # Project dependencies
├── data/
│   └── cyber_laws/       # Folder for source PDF documents
├── db/                   # ChromaDB storage (generated after running loader)
└── src/
    ├── data_loader.py    # Script to ingest PDFs and create embeddings
    └── retrieve.py       # RAG logic, Groq API integration, and prompt building
```

## ⚡ Installation & Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd CyberSense
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the root directory and add your Groq API key:
```ini
GROK_API_KEY=your_groq_api_key_here
GROK_MODEL=llama-3.1-8b-instant
```

## 🏃 Usage

### Step 1: Build the Knowledge Base
Before running the app for the first time (or after adding new PDFs), you must process the documents. Place your legal PDFs in `data/cyber_laws/` and run:

```bash
python src/data_loader.py
```
*This will read PDFs, split text into chunks, generate embeddings, and store them in the `db/` folder.*

### Step 2: Run the Application
Launch the Streamlit interface:

```bash
streamlit run app.py
```

## 🛡️ Disclaimer
**CyberSense** is an AI tool for informational purposes only. It is **not** a substitute for professional legal advice. Users should always consult with a qualified attorney for legal concerns.
