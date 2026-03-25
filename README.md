# RAG Chatbot with Google Generative AI

A robust Retrieval-Augmented Generation (RAG) chatbot designed to chat with multiple PDF documents simultaneously using Google's Gemini models and FastAPI/Streamlit integration. It uses the `gemini-2.5-flash` language model and `gemini-embedding-001` for accurate, context-aware information retrieval.

## Features

- **Multi-PDF Support:** Upload and process multiple PDF files at the same time.
- **Google GenAI Integration:** Leverages cutting-edge Google Gemini LLM for precise question-answering.
- **FAISS Vector Store:** Fast and efficient localized similarity search for document embeddings.
- **Conversation History:** Tracks your conversation context across multiple questions, and allows you to download your chat history as a CSV file.
- **Dynamic Streamlit UI:** Features a fluid, easy-to-use left-hand sidebar menu, custom-styled Chat Messages without dark mode clashes, and real-time processing indicators.

## Prerequisites

Before starting, ensure you have the following installed:
- Python 3.10 or higher
- `pip` package manager
- A [Google API Key](https://ai.google.dev/) initialized for your Generative Language API project.

## Installation Guide

Follow these steps to set up the project locally:

1. **Clone the repository (if applicable)**
   ```bash
   git clone <your-repo-link>
   cd rag_chat_bot
   ```

2. **Create and Activate a Virtual Environment**
   It's highly recommended to use a virtual environment to manage dependencies locally.
   ```bash
   # Create virtual environment
   python3 -m venv venv
   
   # Activate it (Linux/MacOS)
   source venv/bin/activate
   
   # Activate it (Windows PowerShell)
   .\venv\Scripts\Activate.ps1
   ```

3. **Install Dependencies**
   Install all the required Python packages from the `requirements.txt`.
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

Because this application relies on Streamlit to render its User Interface, you **must** use the `streamlit run` command (do not use default `python main.py`, it will not work).

Make sure your virtual environment is still activated:

```bash
# Start the Streamlit application
streamlit run main.py
```

The terminal will provide a local URL (e.g., `http://localhost:8501`). Open this URL in your browser to access the application.

## How to Use

1. **Sidebar Controls:** Enter or confirm your Google API Key if prompted.
2. **Upload Documents:** Under the Menu, upload one or more PDF documents you wish to query.
3. **Submit & Process:** Click the "Submit & Process" button to generate text embeddings and save the FAISS vector index locally.
4. **Chat!:** Use the text input box at the bottom to ask questions specific to the uploaded documents. The chatbot will only answer based strictly on the provided context!
5. **Manage History:** You can clear your session by clicking "Reset" or download the history to a CSV file when needed.

## Troubleshooting

- **404 Embedding NOT FOUND Error:** If you face model existence errors, double-check your Google API Key's endpoint permissions. This bot defaults to `models/gemini-embedding-001` or `models/text-embedding-004` based on API capability.
- **Streamlit ScriptRunContext Missing:** Always run the app using `streamlit run main.py`. Executing it with `python main.py` causes threading conflicts and will not load the UI.
- **Pydantic Validation Errors:** Our repository supports the latest `langchain-google-genai` parameters cleanly. Ensure your pip dependencies match the `requirements.txt`.

---

## Author

Developed by **Anurag Kumar Yadav**

- 📧 **Email:** mranurag101@gmail.com
- 💼 **LinkedIn:** [Anurag Kumar Yadav](https://www.linkedin.com/in/anurag-kumar-yadav-b3b861125)
- 🐙 **GitHub:** [@onurag07](https://github.com/onurag07)
