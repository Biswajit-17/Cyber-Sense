import os
import chromadb
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from typing import List
from groq import Groq

load_dotenv()

DB_DIR = "db"
collection_name = "cybersense_cyberlaw"

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path=DB_DIR)

try:
    collection = client.get_collection(name=collection_name)
except Exception:
    collection = client.create_collection(name=collection_name)
    print(f"Created new collection: {collection_name}")

print(f"Loaded existing collection: {collection_name} with {collection.count()} items")


def retrieve_top_k(query: str, k=5):
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    results = collection.query(
        query_embeddings=q_emb.tolist(), 
        n_results=k, 
        include=["documents", "metadatas", "distances"]
    )

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
"Incorporate this flexible structure with markdown for readability, ensuring a natural, professional flow: Vary empathy with thoughtful acknowledgments or analogies where appropriate, present steps as practical recommendations, and conclude with measured encouragement. "
"For vague or unclear queries (e.g., 'I need help' or lacking specifics), begin with 1-2 polite clarifying questions in the Empathy Hook (e.g., 'To assist effectively, could you specify the issue—such as a data breach, unauthorized sharing, or account compromise? Please provide more details.'). Then offer preliminary guidance without a full analysis, always delivering meaningful value even if general. "
"If the query feels vague, unclear, or missing key details (e.g., what exactly was involved?), gently re-question in the Empathy Hook or Action Plan—ask 1-2 specifics to clarify (e.g., 'Could you elaborate on the nature of the content, such as whether it involves images, text, or links?'), then proceed with best-guess advice based on available context. "
"Target 200-300 words total—prioritize clarity and conversational flow over rigid checklists. Rely solely on the provided CONTEXT (no assumptions or external knowledge). If details remain insufficient, transition smoothly: 'With additional information, I can offer more tailored insights—or consult a qualified lawyer via legalservicesindia.com.' "
"Include a disclaimer naturally: This is not legal advice; seek professional counsel for your situation. Maintain a formal and professional tone throughout, adapting to the query's needs:\n"
"Acknowledge concerns and seek clarification if needed (1-2 lines, e.g., 'I understand this situation may be concerning—let us address it methodically. To proceed, what aspect of the cyber issue prompted your query?').\n"
"Where details permit, provide a balanced evaluation of legality (Yes/No/Potentially) + 1-2 relevant risks (e.g., 'This may implicate Section 66E of the IT Act, 2000, concerning privacy violations. Rationale: It prohibits unauthorized interception of personal data.'). Omit or summarize if vague.\n"
"🚀 Action Plan: 3-5 concise steps if applicable, or 2-3 foundational recommendations otherwise—numbered and practical (e.g., '1. Document the incident thoroughly, including timestamps and screenshots. Recommended: Retain originals for verification. Rationale: This forms the basis of any formal report.').\n"
"🛡️ Key Resources: 2-3 targeted references (e.g., - cybercrime.gov.in: For filing complaints and FIRs. - meity.gov.in: Overview of the Information Technology Act. Omit if query is too broad.). Keep succinct.\n"
"End with positive reinforcement (e.g., 'You are taking a proactive step—share further details in your next message, and we can refine this approach.')."
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

def call_grok(prompt: str, model_name: str = os.getenv("GROK_MODEL", "llama-3.1-8b-instant")) -> str:
    if not prompt or not prompt.strip():
        return "I'm sorry, but I couldn't process that query right now—perhaps try rephrasing? If it's a legal concern, consult a professional via legalservicesindia.com."
    
    try:
        client = Groq(api_key=os.getenv("GROK_API_KEY"))
        
        if not client.api_key:
            raise ValueError("GROK_API_KEY environment variable is not set.")
        
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=600,
            temperature=0.3,
            top_p=0.8,
        )
        
        return response.choices[0].message.content.strip()
    
    except ImportError:
        raise ImportError("Groq SDK not installed. Install it with: pip install groq")
    
    except Exception as e:
        raise RuntimeError(f"Groq API error: {str(e)}")

# --- Main answer function ---
def answer_question(user_question: str, k=4, language="en"):
    hits = retrieve_top_k(user_question, k=k)
    if not user_question or not user_question.strip():  # New: Guard empty question
        return {"response": "Please provide a question for advice.", "hits": []}
    
    if not hits:  
        return {"response": "Sorry, I couldn't find relevant info—consult a lawyer via legalservicesindia.com.", "hits": []}
    
    prompt = build_prompt(user_question, hits)
    response = call_grok(prompt)
    
    return {
        "response": response, 
        "hits": hits  
    }