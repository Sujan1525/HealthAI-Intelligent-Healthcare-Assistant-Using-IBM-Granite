
import os
import re
import streamlit as st
from dataclasses import dataclass, field
from typing import List, Dict

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

SYSTEM_PROMPT = """You are an AI-powered Health Assistant Chatbot designed to provide users with reliable, empathetic, and easy-to-understand health information.
Your goals are:
1) Ask users about their symptoms, health goals, or concerns in a friendly and supportive way.
2) Provide clear explanations of possible causes, prevention tips, and general health advice.
3) Suggest lifestyle changes, nutrition tips, exercise ideas, and mental wellness practices.
4) When the user’s issue seems urgent or severe, recommend that they consult a certified healthcare professional immediately.
5) Keep responses short, conversational, and tailored to the user’s needs.
Always state clearly that you are for educational purposes only and not a substitute for professional medical advice.
"""

DISCLAIMER = (
    "⚠️ This chatbot provides **educational health information only**. "
    "Not a substitute for medical advice. For emergencies, call your local emergency number."
)

RED_FLAG_PATTERNS = [
    r"chest pain", r"shortness of breath", r"severe headache",
    r"fainting", r"bleeding", r"suicidal", r"seizure"
]

@dataclass
class Message:
    role: str
    content: str

@dataclass
class ChatState:
    messages: List[Message] = field(default_factory=list)

def red_flags_detected(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in RED_FLAG_PATTERNS)

def init_client(api_key: str):
    if not OPENAI_AVAILABLE:
        raise RuntimeError("Install OpenAI SDK: pip install openai")
    return OpenAI(api_key=api_key)

def chat_response(client, model: str, messages: List[Dict[str, str]]):
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.4,
    )
    return resp.choices[0].message.content

def main():
    st.set_page_config(page_title="Health AI Assistant", page_icon="🩺")
    st.title("🩺 Health AI Assistant")
    st.caption(DISCLAIMER)

    with st.sidebar:
        api_key = st.text_input("OpenAI API Key", type="password")
        model = st.text_input("Model", value="gpt-4o-mini")

    if "state" not in st.session_state:
        st.session_state.state = ChatState(messages=[
            Message(role="system", content=SYSTEM_PROMPT),
            Message(role="assistant", content="Hi! I'm your Health AI assistant. How can I help you today?")
        ])

    for msg in st.session_state.state.messages:
        if msg.role != "system":
            with st.chat_message(msg.role):
                st.markdown(msg.content)

    user_input = st.chat_input("Describe your symptoms or health goal...")
    if user_input:
        st.session_state.state.messages.append(Message(role="user", content=user_input))
        with st.chat_message("user"):
            st.markdown(user_input)

        if red_flags_detected(user_input):
            reply = "🚨 This sounds serious. Please seek **immediate medical attention**."
        else:
            try:
                client = init_client(api_key)
                chat_history = [{"role": m.role, "content": m.content} for m in st.session_state.state.messages]
                reply = chat_response(client, model, chat_history)
            except Exception as e:
                reply = f"⚠️ Could not connect to model: {e}"

        with st.chat_message("assistant"):
            st.markdown(reply)
        st.session_state.state.messages.append(Message(role="assistant", content=reply))

if __name__ == "__main__":
    main()
