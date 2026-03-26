# 🤖 RAG Chatbot: PDF Q&A with Gemini & FAISS

An end-to-end Retrieval-Augmented Generation (RAG) Chatbot application that allows users to upload PDF documents and ask questions about them. Built with FastAPI for the backend API and Streamlit for the visual chat interface.

The bot uses the Google GenAI Gemini model for answering questions and Google's embedding models for text vectorization, stored locally in a FAISS index.

## 🌟 Features
- **Upload & Parse PDFs**: Easily extract and chunk text from any PDF document.
- **Conversational Memory**: The bot remembers the last 5 turns of your conversation for contextual follow-up questions.
- **Modern RAG Architecture**: Uses LangChain (v1.x classic components like `ConversationalRetrievalChain`), FAISS vector store, and Google Gemini models.
- **Interactive Web Interface**: A clean, responsive Streamlit UI for uploading files and chatting.
- **Source Citations**: Answers include page numbers and text snippets from the original document so you can verify the information.

---

## 🛠️ Technology Stack
- **Backend API**: [FastAPI](https://fastapi.tiangolo.com/), Uvicorn
- **Frontend / UI**: [Streamlit](https://streamlit.io/)
- **LLM & Embeddings**: [Google Generative AI](https://aistudio.google.com/) (Gemini 2.5 Flash, Gemini Embedding 001)
- **Vector Database**: [FAISS](https://github.com/facebookresearch/faiss) (Local CPU installation)
- **Orchestration**: [LangChain](https://python.langchain.com/) / LangChain Classic

---

## 🚀 From Zero to Full Setup

Follow these steps to get the application running locally on your machine.

### 1. Prerequisites
- **Python 3.10+** installed on your system.
- A **Google Gemini API Key**. You can get one for free from [Google AI Studio](https://aistudio.google.com/).

### 2. Clone the Repository & Setup Environment
Open your terminal and create a virtual environment:
```bash
# Clone the repository (if applicable)
git clone <repository_url>
cd rag_chat_bot

# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
# On Linux/macOS:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate
```

### 3. Install Dependencies
Install all the required Python packages from the `requirement.txt` file:
```bash
pip install -r requirement.txt
```

### 4. Setup Environment Variables
Create a file named `.env` in the root directory (where `main.py` is located) and add your Google API Key:
```env
GOOGLE_API_KEY=your_google_api_key_here
```

### 5. Start the Backend Server (FastAPI)
Run the backend web server with `uvicorn`:
```bash
uvicorn app.main:app --reload --port 8000
```
*The API will be available at `http://localhost:8000`.*

### 6. Start the Frontend App (Streamlit)
Open a **new terminal window** (don't forget to activate your virtual environment `source venv/bin/activate` again!), and run the visual interface:
```bash
streamlit run frontend/streamlit_app.py
```
*Your browser will automatically open the web UI at `http://localhost:8501`, where you can upload your PDFs and start chatting!*

---

## 📁 Project Structure
```text
rag_chat_bot/
├── app/
│   ├── main.py           # FastAPI backend endpoints (/upload, /ask)
│   ├── rag_pipeline.py   # PDF loading, chunking, and FAISS vector store logic
│   ├── chains.py         # LangChain LLM setup and Q&A chain creation
│   └── memory.py         # Conversational buffer memory to track history
├── frontend/
│   └── streamlit_app.py  # Visual web UI using Streamlit
├── data/                 # Temporary storage for uploaded PDFs
├── faiss_index/          # Local storage for the generated vector database
├── requirement.txt       # All Python dependencies
└── .env                  # Environment variables (API Keys)
```

---

## 👨‍💻 Author & Details

**Anurag Kumar Yadav**  
Backend Developer transitioning into AI Engineering. Passionate about building intelligent, agentic systems, automating workflows, and multi-agent architectures using modern tech stacks.

🔗 **Connect with me on LinkedIn:** [Anurag Kumar Yadav](https://www.linkedin.com/in/anurag-kumar-yadav-b3b861125)
