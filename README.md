# Cyber-Sense
Features

Adaptive Responses: Analyzes user queries to deliver tailored advice—assesses legality, suggests action plans, and provides resources without rigid templates.
RAG-Powered Accuracy: Retrieves relevant Indian cyber law documents (e.g., IT Act sections, MHA guidelines) from a local ChromaDB vector store for hallucination-free outputs.
User-Friendly Chat Interface: Streamlit-based UI with dark/light mode toggle, bordered chat container, and seamless conversation flow.
Professional Tone: Formal yet reassuring language, with clarifying questions for vague inputs.
Privacy-Focused: All processing is local (embeddings via SentenceTransformers); no external data sharing.

Tech Stack

Frontend: Streamlit (Python web app framework)
Backend:

ChromaDB (vector database for RAG)
SentenceTransformers (embeddings with all-MiniLM-L6-v2)
Google Gemini API (LLM for generation)


Environment: Python 3.10+, dotenv for API keys
Other: No external dependencies beyond listed libs; runs offline post-indexing.

Installation

Clone the Repository:
textgit clone https://github.com/yourusername/cybersense.git
cd cybersense

Set Up Virtual Environment:
textpython -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install Dependencies:
textpip install -r requirements.txt
requirements.txt contents:
textstreamlit
chromadb
sentence-transformers
google-generativeai
python-dotenv

Configure Environment:

Create a .env file in the root:
textGEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.0-flash  # Optional

Obtain a free Gemini API key from Google AI Studio.


Index Documents (One-Time Setup):

Prepare your cyber law PDFs (e.g., IT Act PDF, MHA manuals) in a docs/ folder.
Run the indexing script:
textpython src/index.py  # Assumes you have an index.py for chunking/embedding

This populates the db/ ChromaDB collection named cybersense_cyberlaw.



Usage

Run the App:
textstreamlit run src/app.py

Opens at http://localhost:8501.
Toggle dark mode in the sidebar.
Chat via the input box—e.g., "Is sharing edited photos legal?" or "I need help with a hacked account."


Example Queries:

"Someone posted my edited picture online—what do I do?"
"I posted sensitive content; is this a violation under IT Act?"
Vague: "I need help" → Bot clarifies and offers starters.


Customization:

Edit src/retrieve.py for SYSTEM_INSTRUCTIONS (prompt engineering).
Add more docs to docs/ and re-index for expanded knowledge.



Project Structure
textcybersense/
├── .env                  # API keys
├── .streamlit/config.toml # Theme (dark mode default)
├── db/                   # ChromaDB storage
├── docs/                 # Input PDFs for indexing
├── src/
│   ├── app.py            # Streamlit UI
│   ├── retrieve.py       # RAG + Gemini logic
│   └── index.py          # Document indexing (if separate)
├── requirements.txt
└── README.md
Contributing

Fork the repo and create a feature branch (git checkout -b feature/amazing-feature).
Commit changes (git commit -m 'Add amazing feature').
Push to the branch (git push origin feature/amazing-feature).
Open a Pull Request.

Suggestions: Expand DB with more laws (e.g., DPDP Act), add multi-language support, or integrate voice mode.
License
MIT License—feel free to use, modify, and distribute. See LICENSE for details.
