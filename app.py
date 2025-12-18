import streamlit as st
from src.retrieve import answer_question, build_prompt, call_grok  # Import for hits, prompt, and generation
from dotenv import load_dotenv

load_dotenv()  # Load .env for keys

st.set_page_config(page_title="CyberSense", layout="centered")

# Session state for history and theme
if "messages" not in st.session_state:
    st.session_state.messages = []

# Header row: Title left, toggle right
header_col1, header_col2 = st.columns([8, 1])
with header_col1:
    st.title("🤖 CyberSense — Cyber Helper")

# Bordered Chat Area
st.markdown("<div class='chat-border'>", unsafe_allow_html=True)

# Display history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

st.markdown("</div>", unsafe_allow_html=True)

# Chat Input (Enter to send, no button)
if prompt := st.chat_input("Ask a cyber law question"):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Fixed k=10 (as many as it can)
    k = 10

    # History context for RAG
    history_context = "\n\nPrevious conversation:\n" + "\n".join([
        f"{msg['role'].upper()}: {msg['content']}" for msg in st.session_state.messages[:-1]
    ]) if len(st.session_state.messages) > 1 else ""

    final_q = prompt

    # Call RAG with history
    with st.spinner("Thinking..."):
        try:
            hits = answer_question(final_q, k=k)["hits"]  # Get hits only
            full_prompt = build_prompt(final_q, hits) + history_context  # Inject history
            answer = call_grok(full_prompt)  # Generate

            if not answer.strip():
                answer = "No answer generated—try rephrasing your question."
            st.session_state.messages.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.markdown(answer)

        except Exception as e:
            error_msg = "An error occurred while generating the answer. Check Gemini API config or try again."
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            with st.chat_message("assistant"):
                st.markdown(error_msg)
            st.caption(f"Details: {str(e)}")