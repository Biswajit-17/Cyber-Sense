import os
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from typing import List

load_dotenv()

# Settings
DB_DIR = "db"
collection_name = "cybersense_cyberlaw"  # Ensure this matches your indexing script!

# Load local embedding model (MUST match the one used in indexing: "all-MiniLM-L6-v2")
embed_model = SentenceTransformer("all-MiniLM-L6-v2")

# Chroma client (modern method: PersistentClient for automatic persistence)
client = chromadb.PersistentClient(path=DB_DIR)

# Get or create collection (with error handling)
try:
    collection = client.get_collection(name=collection_name)
except Exception:
    # If collection doesn't exist, create it (adjust metadata if needed)
    collection = client.create_collection(name=collection_name)
    print(f"Created new collection: {collection_name}")

print(f"Loaded existing collection: {collection_name} with {collection.count()} items")

# --- Retrieval function ---
def retrieve_top_k(query: str, k=5):
    # Encode query (convert_to_numpy=True for compatibility)
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    results = collection.query(
        query_embeddings=q_emb.tolist(), 
        n_results=k, 
        include=["documents", "metadatas", "distances"]
    )
    # Extract results
    docs = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]
    hits = []
    for doc, meta, dist in zip(docs, metadatas, distances):
        hits.append({"text": doc, "meta": meta, "score": dist})
    return hits

# --- Prompt builder ---
SYSTEM_INSTRUCTIONS = (
"You are CyberSense, a reliable cyber law advisor for India—providing clear, professional guidance with empathy and precision. Use formal, straightforward language that is accessible and reassuring, avoiding unnecessary complexity."
"Analyze the user's query first: Determine its intent (e.g., seeking legal status, steps for action, resources, clarification, or general info). Tailor your response accordingly—use a natural, adaptive structure that fits the ask. For example:"
"- If it's a direct legal question (e.g., 'Is X legal?'), lead with a concise assessment + risks, then steps if relevant."
"- For advice on 'what to do', prioritize an action plan with 2-4 steps."
"- For vague/broad queries (e.g., 'I need help'), start with 1-2 clarifying questions, then offer preliminary guidance."
"- Keep it conversational yet professional: Acknowledge concerns upfront, weave in resources naturally, and end with encouragement."
"Incorporate markdown sparingly for readability (e.g., bold for key points, numbered lists for steps)—but only where it enhances clarity, not as a rigid template."
"Target 150-250 words—prioritize relevance and brevity. Rely solely on the provided CONTEXT (no assumptions or external knowledge). If details are insufficient, note it politely and suggest: 'For more precision, provide additional specifics—or consult a qualified lawyer via legalservicesindia.com.'"
"Include a disclaimer naturally: This is not legal advice; seek professional counsel for your situation. Maintain a professional tone throughout, adapting fully to the query's needs."
)
def build_prompt(user_question: str, retrieved: List[dict]):
    context_texts = [item['text'] for item in retrieved]  # Just texts, no sources
    context_block = "\n\n---\n\n".join(context_texts)
    prompt = (
        f"{SYSTEM_INSTRUCTIONS}\n\n"
        f"CONTEXT:\n{context_block}\n\n"
        f"USER QUESTION: {user_question}\n\n"
        f"Respond in the exact structure—use markdown for Streamlit rendering."
    )
    print(f"DEBUG: Built prompt length: {len(prompt)}, preview: {prompt[:100]}...")
    return prompt

# --- Gemini call (implement with Google Generative AI) ---
def call_gemini(prompt: str, model_name: str = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")):
    if not prompt or not prompt.strip():
        return "I'm sorry, but I couldn't process that query right now—perhaps try rephrasing? If it's a legal concern, consult a professional via legalservicesindia.com."
    try:
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel(
            model_name,
            generation_config={
                "max_output_tokens": 600,  # Room for full structure
                "temperature": 0.3,  # Low for consistent, grounded responses
                "top_p": 0.8,  # Balanced creativity
                "top_k": 40
            },
            safety_settings={
                "HARM_CATEGORY_HARASSMENT": "BLOCK_MEDIUM_AND_ABOVE",
                "HARM_CATEGORY_HATE_SPEECH": "BLOCK_MEDIUM_AND_ABOVE",
                # Add more for legal sensitivity if needed
            }
        )
        response = model.generate_content(contents=[prompt])
        return response.text
    except ImportError:
        raise ImportError("Install google-generativeai: pip install google-generativeai")
    except Exception as e:
        raise RuntimeError(f"Gemini API error: {e}")

    # --- Simple fallback echo during development (uncomment if needed) ---
    # return "GENERATION_PLACEHOLDER: " + prompt[:500]  # debug

# --- Main answer function ---
def answer_question(user_question: str, k=4, language="en"):
    hits = retrieve_top_k(user_question, k=k)
    if not user_question or not user_question.strip():  # New: Guard empty question
        return {"response": "Please provide a question for advice.", "hits": []}
    
    if not hits:  # Fallback if no retrieval
        return {"response": "Sorry, I couldn't find relevant info—consult a lawyer via legalservicesindia.com.", "hits": []}
    
    prompt = build_prompt(user_question, hits)
    response = call_gemini(prompt)
    
    return {
        "response": response,  # Full generated text for rendering
        "hits": hits  # For citations/sources in app
    }
# --- Example usage (uncomment to test) ---
'''if __name__ == "__main__":
    question = "I posted *** is this legal if not then what should i do and what am i violating"  # Your sample
    result = answer_question(question)
    print("Full Response:\n", result["response"])
    print("\nHits/Sources:\n", result["hits"])'''