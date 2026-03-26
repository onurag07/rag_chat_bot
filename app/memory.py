# memory.py — Store chat history per user session

from langchain_classic.memory import ConversationBufferWindowMemory
from langchain_classic.chains import ConversationalRetrievalChain
from app.chains import get_llm

# Dictionary to store one memory object per user session
# In production, use Redis instead of a dict (survives server restarts)
session_memories = {}


def get_memory(session_id: str):
    """
    Get or create memory for a session.

    ConversationBufferWindowMemory(k=5):
    - k=5 means remember the last 5 turns (5 questions + 5 answers)
    - Older turns are dropped automatically
    - This prevents context overflow on long conversations
    """
    if session_id not in session_memories:
        session_memories[session_id] = ConversationBufferWindowMemory(
            k=5, memory_key="chat_history", return_messages=True, output_key="answer"
        )
    return session_memories[session_id]


def build_chat_chain(retriever, session_id: str):
    """
    Build a conversational chain with memory.
    This is the chain to use for the chat interface.
    """
    memory = get_memory(session_id)
    llm = get_llm()

    return ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=retriever, memory=memory, return_source_documents=True
    )


def clear_memory(session_id: str):
    """Clear memory for a session (user clicks 'Clear chat')"""
    if session_id in session_memories:
        del session_memories[session_id]
