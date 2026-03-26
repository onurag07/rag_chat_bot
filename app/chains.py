from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()


def get_llm(temperature=0.2):
    """
    Create the Gemini LLM instance.

    temperature controls randomness:
      0.0 = very factual, deterministic (same answer every time)
      1.0 = very creative, varied (different answer each time)
      0.2 = recommended for Q&A — mostly factual but slightly natural
    """
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=temperature,
    )


def build_qa_chain(retriever, chain_type="stuff"):
    llm = get_llm()
    prompt_template = """
    You are a helpful assistant that answers questions based ONLY on the provided document context.

    Context from the document:
    {context}

    Question: {question}

    Instructions:
    - Answer ONLY using the information in the context above
    - If the answer is not in the context, say "I couldn't find this information in the document."
    - Be concise and clear
    - Cite the page number if possible

    Answer:
    """
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type=chain_type,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": prompt,
        },
    )

    return qa_chain


def ask_question(chain, question: str):
    """
    Ask a question and return the answer + sources.

    Returns dict with:
      - answer: the text answer from Gemini
      - sources: list of {page, text_snippet} used to form the answer
    """
    try:
        result = chain.invoke({"query": question})
        sources = []
        for doc in result.get("source_documents", []):
            sources.append(
                {
                    "page": doc.metadata.get("page", "Unknown"),
                    "text": doc.page_content[:150] + "...",
                }
            )
        return {"answer": result["result"], "sources": sources}
    except Exception as e:
        return {"answer": f"Error: {str(e)}", "sources": []}


def build_summarise_chain(retriever, method="map_reduce"):
    """
    Build a chain for summarising long documents.

    method="map_reduce" → fast, parallel, good for very long docs
    method="refine"     → slow, sequential, best quality

    Use chain_type= to switch between them — LangChain handles the rest!
    """
    llm = get_llm(temperature=0.3)
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type=method,  # "map_reduce" or "refine"
        retriever=retriever,
        return_source_documents=True,
    )
