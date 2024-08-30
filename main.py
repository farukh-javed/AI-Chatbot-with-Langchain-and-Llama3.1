from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_ollama import OllamaLLM
import streamlit.components.v1 as components
import streamlit as st
import requests

def load_lottie_url(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except requests.exceptions.RequestException:
        return None

def st_lottie(lottie_json, height=None, key=None):
    if lottie_json is not None:
        lottie_html = f'''
        <script src="https://cdnjs.cloudflare.com/ajax/libs/bodymovin/5.7.4/lottie.min.js"></script>
        <div id="lottie-{key}"></div>
        <script>
        var animation = lottie.loadAnimation({{
            container: document.getElementById('lottie-{key}'),
            renderer: 'svg',
            loop: true,
            autoplay: true,
            animationData: {lottie_json}
        }});
        </script>
        '''
        st.components.v1.html(lottie_html, height=height)
    else:
        st.warning("Lottie animation could not be loaded. Please check your internet connection or the animation URL.")

robot_animation = load_lottie_url("https://lottie.host/00401cd3-d8f0-471d-8cc0-87537459ce56/TocVglSQ9o.json")
typing_animation = load_lottie_url("https://lottie.host/ae30fcf4-3178-4e6f-ad41-63660666dc2b/dQ6VvbSAjl.json")

st.set_page_config(
    page_title="Interactive AI Chatbot",
    page_icon="🤖",
    layout="centered",
)

st.sidebar.title("🔧 Settings")
language = st.sidebar.selectbox("🌐 Select Language", options=["English", "Urdu", "Spanish", "French"])
temperature = st.sidebar.slider("🔥 Temperature", min_value=0.0, max_value=1.0, value=0.5)
reset_button = st.sidebar.button("🔄 Reset Conversation")

if reset_button or "conversation" not in st.session_state:
    llm = OllamaLLM(model="llama3.1:8b", temperature=temperature)
    memory = ConversationBufferMemory()
    st.session_state.conversation = ConversationChain(llm=llm, memory=memory, verbose=True)
    st.session_state.messages = []

conversation = st.session_state.conversation

st.title("💬 AI-Powered Chatbot")
st_lottie(robot_animation, height=150, key="robot")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

prompt = st.chat_input("Type your message here...")

if prompt:
    if prompt.lower() == 'exit':
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    st_lottie(typing_animation, height=50, key="typing")

    response = conversation.predict(input=prompt)
    st.session_state.messages.append({"role": "assistant", "content": response})

    with st.chat_message("assistant"):
        st.write(response)

with st.expander("🛠 Customize Chatbot"):
    st.write("Adjust settings to enhance your experience.")
    st.slider("Response Speed", min_value=0.0, max_value=1.0, value=0.5)
    st.radio("Tone of Response", options=["Formal", "Casual", "Friendly"], index=1)
    st.checkbox("Enable Emojis", value=True)
