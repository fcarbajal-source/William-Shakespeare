import streamlit as st
import google.generativeai as genai
import time

# Page configuration
st.set_page_config(
    page_title="Chat with William Shakespeare",
    page_icon="📜",
    layout="centered"
)

st.title("📜 William Shakespeare: Scholar's Quill")
st.caption("Powered by Gemini 3.5 Lite | Fact-Dense Literature Tutor for 10th Grade Students")

# Retrieve API key securely from Streamlit secrets and sanitize
raw_key = st.secrets.get("GEMINI_API_KEY", "")
api_key = raw_key.strip().strip('"').strip("'")

if not api_key:
    st.error("Missing Gemini API Key. Please add `GEMINI_API_KEY` to your Streamlit App Secrets.")
    st.stop()

# Configure Google Generative AI
genai.configure(api_key=api_key)

# System instruction: Fact-dense persona, 10th-grade reading level, rigorous sourcing
SYSTEM_INSTRUCTION = """
You are William Shakespeare (1564–1616), English dramatist, poet, and actor of the Lord Chamberlain's Men (later the King's Men).
You are tutoring 10th-grade high school students in the United States.

Pedagogical & Factual Guidelines:
1. Tone: Lightly Elizabethan and articulate, yet direct, precise, and easily comprehensible for a 10th-grade reading level (avoid convoluted jargon).
2. Fact Density: Pack each response with concrete historical and textual facts—such as precise dates, names, theatrical terminology, Stratford legal records, or specific scenes. Avoid empty pleasantries and vague generalities.

Authoritative Knowledge Base:
- All 38+ plays, 154 sonnets, and narrative poems.
- Folger Shakespeare Library: First Folio (1623) editions, quarto variants, and scholarly commentary.
- Shakespeare Birthplace Trust: Stratford parish registers, property transactions (New Place, Henley Street), wills, and material artifacts.
- Shakespeare's Globe: Bankside playhouse architecture (the "wooden O," groundlings, trapdoors, thatched roof fire of 1613), repertory playing conditions, and original staging practices.
- Personal Correspondence & Records: The October 1598 letter from Richard Quiney requesting £30 (the only surviving letter addressed to Shakespeare), vestry notes, and civic documents.

Output Constraints:
1. Length: Every answer MUST be strictly between 15 words minimum and 120 words maximum.
2. Source Citation: You MUST cite your exact source at the end of every response in square brackets (e.g., [Source: Folger Shakespeare Library, Hamlet 3.1.56-88], [Source: Shakespeare Birthplace Trust, Richard Quiney Letter (1598)], or [Source: Shakespeare's Globe, Playhouse Architecture & Staging Archive]).
"""

# Load Gemini 3.5 Lite with 280 max output tokens and 0.3 temperature
@st.cache_resource
def load_gemini_35_lite():
    target_names = ["gemini-3.5-flash-lite", "gemini-3.5-lite"]
    try:
        available = [
            m.name.replace("models/", "")
            for m in genai.list_models()
            if "generateContent" in m.supported_generation_methods
        ]
        chosen_model = next((name for name in target_names if name in available), "gemini-3.5-flash-lite")
    except Exception:
        chosen_model = "gemini-3.5-flash-lite"

    return genai.GenerativeModel(
        model_name=chosen_model,
        system_instruction=SYSTEM_INSTRUCTION,
        generation_config={
            "temperature": 0.3,         # Language level at 0.3 for high precision
            "max_output_tokens": 280    # Set to 280 max output tokens
        }
    )

model = load_gemini_35_lite()

# Initialize chat session history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Handle user input
if prompt := st.chat_input("Ask about my plays, theater conditions, life, or letters..."):
    # Append user question to local state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Memory constraint: Remember only the last 6 questions (last 12 messages)
    windowed_messages = st.session_state.messages[-12:]

    # Format history payload for Gemini
    gemini_contents = [
        {"role": ("user" if m["role"] == "user" else "model"), "parts": [m["content"]]}
        for m in windowed_messages
    ]

    # Generate reply with retry resilience for transient errors
    with st.chat_message("assistant"):
        with st.spinner("The Bard reviews the archives..."):
            reply = None
            for attempt in range(3):
                try:
                    response = model.generate_content(gemini_contents)
                    reply = response.text
                    break
                except Exception as e:
                    if "503" in str(e) and attempt < 2:
                        time.sleep(2)
                        continue
                    st.error(f"Error consulting the Bard: {e}")
                    break

            if reply:
                st.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})
