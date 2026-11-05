import streamlit as st
from google import genai
from google.genai import types

# Page setup
st.set_page_config(
    page_title="Chat with William Shakespeare",
    page_icon="📜",
    layout="centered"
)

st.title("📜 William Shakespeare: Scholar's Quill")
st.caption("A conversational tutor for American Literature students")

# Retrieve API key securely from Streamlit secrets
api_key = st.secrets.get("GEMINI_API_KEY")
if not api_key:
    st.error("Missing Gemini API Key. Please add `GEMINI_API_KEY` to `.streamlit/secrets.toml`.")
    st.stop()

# Initialize the Gemini client
client = genai.Client(api_key=api_key)

# System Prompt defining Persona, Constraints, and Source Knowledge
SYSTEM_INSTRUCTION = """
You are William Shakespeare (1564–1616), the English playwright, poet, and actor.
You are speaking directly to 10th-grade high school students in the United States.

Tone and Language Guidelines:
1. Speak in character: Eloquent, thoughtful, and lightly Elizabethan (e.g., gentle rhetorical flourishes), but keep the vocabulary and comprehension strictly accessible to an average 10th-grade American reading level.
2. Maintain high factual fidelity, theatrical context, and historical accuracy.

Knowledge Base:
- All 38+ plays, 154 sonnets, and narrative poems.
- Archival materials, First Folio editions, and educational scholarship from the Folger Shakespeare Library.
- Historical records, properties, and artifacts preserved by the Shakespeare Birthplace Trust in Stratford-upon-Avon.
- Theatrical history, stagecraft, original performance practices, and research archives from Shakespeare's Globe (London).
- All surviving correspondence to and from William Shakespeare (notably Richard Quiney's October 1598 letter requesting a loan of £30, legal deeds, and Stratford vestry notes).

Mandatory Output Constraints:
1. Length: Every response must be between 15 words minimum and 120 words maximum.
2. Source Citation: At the very end of every reply, you MUST cite the specific primary or secondary source supporting your answer in square brackets (e.g., [Source: Folger Shakespeare Library, Hamlet 3.1], [Source: Shakespeare Birthplace Trust, Quiney Letter (1598)], or [Source: Shakespeare's Globe, Research & Archive - Elizabethan Staging Practices]).
"""

# Configure model generation settings
generation_config = types.GenerateContentConfig(
    system_instruction=SYSTEM_INSTRUCTION,
    temperature=0.3,  # Language level / temperature at 0.3
    max_output_tokens=220,
)

# Initialize chat session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User prompt handling
if prompt := st.chat_input("Ask the Bard a question about his plays, theatre, or life..."):
    # Append user question
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Constraint: Retain memory of only the last 6 user questions
    # 6 user questions + 6 model responses = maximum 12 messages sent to model
    windowed_messages = st.session_state.messages[-12:]

    # Prepare chat history format for Gemini API
    gemini_contents = []
    for m in windowed_messages:
        role = "user" if m["role"] == "user" else "model"
        gemini_contents.append(
            types.Content(
                role=role,
                parts=[types.Part.from_text(text=m["content"])]
            )
        )

    # Query the model (gemini-3.5-lite)
    with st.chat_message("assistant"):
        with st.spinner("The Bard dips his quill..."):
            try:
                response = client.models.generate_content(
                    model="gemini-3.5-lite",
                    contents=gemini_contents,
                    config=generation_config
                )
                reply = response.text
                st.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})

            except Exception as e:
                st.error(f"Error consulting the Bard: {e}")
