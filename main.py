from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq

import streamlit as st

st.set_page_config(page_title="ZeeChatBot", page_icon="🤖")
st.subheader("🌐 :orange[Zee]ChatBot 🤖", divider="violet")


# Set up memory
msgs = StreamlitChatMessageHistory(key="langchain_messages")
if len(msgs.messages) == 0 :
    msgs.add_ai_message("👋 Hello, I'm here to help, How can I assist you today 📚 ? :sparkles:")

# app info
with st.sidebar:
    st.text("💻 Simple ChatBot with Conversation memory Streamlit App built using LangChain and Groq Llama3")
    st.code("github.com/0xZee/streamlit-chatbot")

# Get an Groq API Key before continuing
st.sidebar.subheader("🔐 Groq API Key", divider="violet")
if "GROQ_API" in st.secrets:
    GROQ_API = st.secrets['GROQ_API']
    st.sidebar.success("✅ GROQ API Key Set")
else:
    GROQ_API = st.sidebar.text_input("Groq API Key", type="password")
if not GROQ_API:
    st.info("Enter an Groq API Key to continue (groq.com)")
    st.stop()

st.sidebar.subheader("🤖 LLM Model", divider="violet")
model_name = st.sidebar.selectbox(
    "Select Model (META, MISTRAL, GOOGLE) ",
    ("llama3-8b-8192", "mixtral-8x7b-32768", "gemma-7b-it"))
st.sidebar.divider()

# Set up the LangChain, passing in Message History

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a kind and helpful AI chatbot having a conversation with a human."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)
# llm and chain
llm = ChatGroq(temperature=0.1, groq_api_key=GROQ_API, model_name=model_name)
chain = prompt | llm
chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: msgs,
    input_messages_key="question",
    history_messages_key="history",
)

# Render current messages from StreamlitChatMessageHistory
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

# Clear chat history
if st.sidebar.button("Clear Chat History", use_container_width=True):
    msgs.clear()
    msgs.add_ai_message("👋 Hello, I'm here to help, How can I assist you today 📚 ? :sparkles:")

# chat session history
view_messages = st.sidebar.expander("View Session Conversation History Content")

# If user inputs a new prompt, generate and draw a new response
if prompt := st.chat_input():
    st.chat_message("human").write(prompt)
    # Note: new messages are saved to history automatically by Langchain during run
    config = {"configurable": {"session_id": "any"}}
    response = chain_with_history.invoke({"question": prompt}, config)
    st.chat_message("ai").write(response.content)

# Draw the messages at the end, so newly generated ones show up immediately
with view_messages:
    """
    Message History initialized with:
    ```python
    msgs = StreamlitChatMessageHistory(key="langchain_messages")
    ```

    Contents of `st.session_state.langchain_messages`:
    """
    view_messages.json(st.session_state.langchain_messages)
