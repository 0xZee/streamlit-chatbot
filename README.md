# 🤖 ChatBot with Conversation Memory : Streamlit App, `LangChain`, StreamlitChatMessageHistory, Groq API to : llama3, Mixtral, Gemma


This repository contains an example of a Memero Conversational ChatBot (RAG) application built using `LangChain` and `Groq` Llama3. 

## 🛠️ Components

1. **LangChain**:
   - llama-index provides the core functionality for handling language models, prompts, and text processing.
   - We use the Llama3 LLM (Large Language Model) from llama-index for text generation.

2. **Chroma**:
   - Chroma is used as the vector store for document embeddings.
   - It organizes and indexes documents based on high-dimensional vectors.

3. **Groq Llama3**:
   - Groq Llama3 is integrated for querying and retrieving relevant documents.
   - It combines Groq queries with Llama3 embeddings to fetch contextually relevant information from PDF.


## Usage

1. **Installation**:
   - Install the required Python packages using `pip install -r requirements.txt`.

2. **Configuration**:
   - Set up your Groq API key and other necessary credentials.

3. **Run the RAG System**:
   - Initialize the app with Groq Api on Streamlit App


## To run App
```python
streamlit run main.py


