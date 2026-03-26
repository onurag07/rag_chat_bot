import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from pypdf.errors import PdfReadError
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()


def load_document(path: str, password: str = None):
    """
    Load A Pdf (or Url) and return itspages as alist Document objects.

    Each Documnet objects has:
    - page_content: The text of the page.
    - metadata: A dictionary containing:
        - source: The path to the file.
        - page: The page number.

    Returns: (documnets, Error_message)
        - If Success: (list_of_documents, None)
        - If Error: (None, error_message)
    """
    try:
        if path.endswith(".pdf"):
            loader = PyPDFLoader(path, password=password)
        else:
            loader = WebBaseLoader(path)
        documents = loader.load()

        total_text = " ".join(doc.page_content for doc in documents).strip()

        if not total_text:
            return (
                None,
                "PDF Has no Extractable text. It may be scanned image. try different pdf",
            )

        if len(total_text) < 50:
            return (
                None,
                f"Pdf too short to process ({len(total_text)}) characters found.",
            )

        return documents, None

    except PdfReadError as e:
        if "encrypted" in str(e).lower():
            return None, "PDF is encrypted. Please provide password."
        return None, f"Corruped PDF: {str(e)}"

    except FileNotFoundError:
        return None, f"file not found: {path}"

    except Exception as e:
        raise ValueError(f"Error loading document: {e}")


def chunk_documents(documents, chunk_size, chunk_overlap=200):
    """
    Split documents into smaller chunks.

    chunk_size=1000   → each chunk is ~1000 characters long
    chunk_overlap=200 → each chunk shares 200 chars with the next one

    RecursiveCharacterTextSplitter is "smart" — it tries to split on:
      1. Paragraphs first (keeps ideas together)
      2. Then sentences
      3. Then words (last resort)
    This gives more natural, coherent chunks than just cutting at 1000 chars.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        # chunk.metadata["source"] = documents[0].metadata["source"]
        # chunk.metadata["page"] = documents[0].metadata["page"]
        # chunk.metadata["chunk_number"] = i + 1
        # chunk.metadata["total_chunks"] = len(chunks)

    print(f" split into {len(chunks)} chunks (avg {chunk_size} chars each)")
    return chunks


def get_embeddings():
    """
     Create the embedding model.

    text-embedding-004 is Google's free embedding model.
    It converts text → a list of 768 numbers.
    768 dimensions = it can capture 768 different aspects of meaning.

    IMPORTANT: Always use the SAME embedding model for storing and querying.
    If you store with model A but search with model B, results will be random garbage.
    """
    return GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")


def build_vectorstore(chunks, embeddings, save_path="faiss_index"):
    """
    Convert chunks to vectors and store them in FAISS.

    FAISS.from_documents() does two things at once:
      1. Calls the embedding model on every chunk (API call to Google)
      2. Stores all vectors in a searchable index

    We then save to disk so we never have to re-embed the same PDF again.
    """
    print("🔄 Building FAISS index (calling embedding API for each chunk)...")
    print("   This may take 30-60 seconds for large PDFs — be patient!")

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(save_path)
    print("✅ FAISS index saved to", save_path)
    return vectorstore


def load_vectorstore(embeddings, load_path="faiss_index"):
    """
    Load a previously saved FAISS index from disk.
    Much faster than rebuilding — use this on server restarts.
    """
    vectorstore = FAISS.load_local(
        load_path, embeddings, allow_dangerous_deserialization=True
    )
    print("✅ FAISS index loaded from", load_path)
    return vectorstore


def process_pdf(pdf_path: str, force_rebuild=False):
    """
    MASTER FUNCTION — runs the full Stage 1-4 pipeline.

    1. Load PDF
    2. Chunk it
    3. Create/load embeddings model
    4. Build or load FAISS index

    force_rebuild=True → re-embed even if index already exists
    force_rebuild=False → load from disk if index exists (fast path)
    """

    index_path = "fasiss_index"
    embeddings = get_embeddings()

    if os.path.exists(index_path) and not force_rebuild:
        print("Loading existing FAISS index...(fast!)")
        return load_vectorstore(embeddings, index_path), None

    docs, err = load_document(pdf_path)
    vectorstore = build_vectorstore(docs, embeddings, index_path)
    return vectorstore, None


def get_retriever(vectorstore, strategy="mmr", k=5):
    """
    Create a retriever from the vectorstore.

    strategy options:
      "similarity"  → top-k most similar chunks. Simple. Fast.
      "mmr"         → top-k diverse & relevant chunks. Better answers. (Recommended)
      "threshold"   → only return chunks above a similarity score (filters irrelevant results)

    k = how many chunks to retrieve. 4-5 is the sweet spot.
      Too few → might miss the answer
      Too many → sends too much text to the LLM (costs more, slower)
    """

    if strategy == "mmr":
        return vectorstore.as_retriever(
            search_type="mmr", search_kwargs={"k": k, "fetch_k": 20}
        )
    elif strategy == "similarity":
        return vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": k}
        )
    else:  # "threshold":
        return vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": k, "score_threshold": 0.7}
        )
