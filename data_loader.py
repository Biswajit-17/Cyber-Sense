import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
import chromadb
from chromadb.config import Settings

# Paths
DATA_DIR = "data/cyber_laws"  
DB_DIR = "db"                 

# Initialize ChromaDB (modern method)
client = chromadb.PersistentClient(path="db")

# Create a collection (like a table) for your embeddings
collection_name = "cybersense_cyberlaw"
if collection_name in [c.name for c in client.list_collections()]:
    collection = client.get_collection(name=collection_name)
else:
    collection = client.create_collection(name=collection_name)

# Prepare your embeddings model
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        reader = PdfReader(pdf_path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return text

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)

# Process each PDF and add to Chroma
for filename in os.listdir(DATA_DIR):
    if filename.lower().endswith(".pdf"):
        file_path = os.path.join(DATA_DIR, filename)
        print(f"Processing: {filename}")

        text = extract_text_from_pdf(file_path)
        if not text.strip():
            print(f"Warning: {filename} has no readable text. Skipping.")
            continue

        chunks = text_splitter.split_text(text)
        print(f" → Split into {len(chunks)} chunks.")

        # Add chunks to ChromaDB
        for i, chunk in enumerate(chunks):
            collection.add(
                documents=[chunk],
                metadatas=[{"source": filename, "chunk": i}],
                ids=[f"{filename}_{i}"],
                embeddings=embedding_model.embed_documents([chunk])
            )

print("All PDFs processed and embeddings saved to ChromaDB!")
print(f"Final collection count after adding: {collection.count()}")