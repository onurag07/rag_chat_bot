from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

# Resolve the absolute path of this python file's directory (app folder)
# and go one level up to the root project folder
ROOT_DIR = Path(__file__).resolve().parent.parent
pdf_path = ROOT_DIR / "data" / "pdf" / "Project1_RAG_Chatbot_Implementation_Plan.pdf"

try:
    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()
    
    if not documents:
        print("Warning: The PDF is empty or contains no readable text.")
    else:
        print(f"Loaded {len(documents)} documents")
        print("Metadata of first page:", documents[0].metadata)
        print("Content of first page:", documents[0].page_content[:200], "...") # snippet
        
except Exception as e:
    error_msg = str(e).lower()
    if "password" in error_msg or "encrypt" in error_msg or "decrypt" in error_msg:
        print("Error: The PDF is password-protected or encrypted. Please provide a decrypted version.")
    else:
        print(f"Error loading PDF: {e}")
