# Chat with PDF and DOC using LangGraph + OpenAI

<img src="./art/demo.gif?raw=true">

## Overview:
This project provides a simple and user-friendly chatbot capable of answering questions and extracting information from PDF and DOC files using the OpenAI language model. The RAG pipeline is built as a **LangGraph** state graph (`retrieve` → `generate`), with conversation history persisted through a LangGraph checkpointer.

## Features:
- Interactive chatbot interface
- PDF and DOC parsing
- LangGraph state graph for the RAG flow, with `MemorySaver` checkpointing for multi-turn memory
- FAISS vector store with HuggingFace embeddings
- Easy-to-use with Streamlit web application

## Graph:
```
START -> retrieve (FAISS similarity search) -> generate (ChatOpenAI) -> END
```

## Installation:
1. Clone the repository:
```git clone https://github.com/yourusername/chat-with-pdf-doc.git```
```cd chat-with-pdf-doc```

2. Install dependencies:
```pip install -r requirements.txt```

3. Start the Streamlit app:
```streamlit run app.py```

## License:
This project is licensed under the MIT License.

## Acknowledgments:
- This project uses the [OpenAI API](https://openai.com) for natural language processing.

Feel free to use this project.
