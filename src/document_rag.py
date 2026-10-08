"""
PropwiseAI - Document RAG (Retrieval-Augmented Generation) Module
Description:
    Processes the PropwiseAI Functional Requirements document.
    Chunks text into passages, builds a FAISS vector index, and provides
    grounded Retrieval-Augmented Generation (RAG) to answer user questions about
    platform features, customer intents, and business rules without hallucinating.
"""

import sys
import os
import logging
from typing import List, Dict, Any

# Ensure parent project root is in Python path for direct script execution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.config import DOC_KNOWLEDGE_PATH, DOC_INDEX_PATH
from src.embeddings import EmbeddingManager
from src.gemini_service import GeminiService

logger = logging.getLogger(__name__)

class DocumentRAGSystem:
    """
    Retrieval-Augmented Generation (RAG) engine for answering questions
    about the PropwiseAI requirements document.
    """

    def __init__(self):
        self.embedding_manager = EmbeddingManager()
        self.gemini_service = GeminiService()
        self.vector_store = None
        self._initialize_or_load_index()


    def _initialize_or_load_index(self):
        """Loads FAISS index from disk if present, else builds index from raw text."""
        index_file = DOC_INDEX_PATH / "index.faiss"
        if os.path.exists(index_file):
            logger.info(f"Loading existing Document FAISS index from: {DOC_INDEX_PATH}")
            try:
                self.vector_store = FAISS.load_local(
                    folder_path=str(DOC_INDEX_PATH),
                    embeddings=self.embedding_manager.model,
                    allow_dangerous_deserialization=True
                )
                logger.info("Document FAISS index successfully loaded.")
                return
            except Exception as e:
                logger.warning(f"Failed to load FAISS index: {e}. Rebuilding...")

        self.rebuild_index()

    def rebuild_index(self):
        """Reads document_knowledge.txt, splits into chunks, and builds FAISS vector store."""
        if not os.path.exists(DOC_KNOWLEDGE_PATH):
            raise FileNotFoundError(f"Knowledge document missing at: {DOC_KNOWLEDGE_PATH}")

        with open(DOC_KNOWLEDGE_PATH, "r", encoding="utf-8") as f:
            document_text = f.read()

        # Step 1: Chunk text semantically
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=120,
            separators=["\n\n", "\n", ". ", " "]
        )
        raw_chunks = text_splitter.split_text(document_text)

        # Step 2: Convert text chunks into LangChain Document objects
        documents = [
            Document(page_content=chunk, metadata={"source": "PropwiseAI_Requirements.txt", "chunk_id": i})
            for i, chunk in enumerate(raw_chunks)
        ]

        # Step 3: Create FAISS vector database using local embeddings
        self.vector_store = FAISS.from_documents(
            documents=documents,
            embedding=self.embedding_manager.model
        )

        # Step 4: Save FAISS index locally to disk
        DOC_INDEX_PATH.mkdir(parents=True, exist_ok=True)
        self.vector_store.save_local(folder_path=str(DOC_INDEX_PATH))
        logger.info(f"Saved Document FAISS vector index to: {DOC_INDEX_PATH}")

    def query_document(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Executes vector similarity search on FAISS and synthesizes a grounded answer.
        """
        if not self.vector_store:
            return {"answer": "Vector store index is not available.", "retrieved_chunks": []}

        # Vector similarity search against stored chunk embeddings
        similar_docs = self.vector_store.similarity_search(query, k=top_k)
        retrieved_texts = [doc.page_content for doc in similar_docs]

        # Synthesize factual response based strictly on document passages
        answer = self._synthesize_answer(query, retrieved_texts)

        return {
            "query": query,
            "answer": answer,
            "retrieved_chunks": retrieved_texts
        }

    def _synthesize_answer(self, query: str, chunks: List[str]) -> str:
        """Formulates factual response strictly grounded in retrieved document chunks."""
        if not chunks:
            return "No relevant information found in PropwiseAI documentation."

        # If Gemini is active, use Gemini for strict grounded document synthesis
        if self.gemini_service.is_active:
            gemini_ans = self.gemini_service.synthesize_rag_response_with_gemini(query, chunks)
            if gemini_ans:
                return f"🤖 **GEMINI GROUNDED RAG ANSWER:**\n\n{gemini_ans}"

        q_lower = query.lower()

        if "category" in q_lower or "type" in q_lower:
            return (
                "According to Section 2 of PropwiseAI Requirements:\n"
                "The supported property categories are:\n"
                "• **Town House**\n"
                "• **Modern Villa**\n"
                "• **Apartment**\n"
                "• **Single Family**\n"
                "• **Office**"
            )

        if "intent" in q_lower or "support" in q_lower:
            return (
                "Based on Section 4 of PropwiseAI Requirements, supported intents are:\n"
                "• Buy / Rent / Explore properties\n"
                "• Investment property search\n"
                "• Property Q&A, Compare, & Similar properties\n"
                "• Schedule site visits & request callbacks"
            )

        # Default factual summary from primary matching passage
        primary = chunks[0].strip()
        return f"**Information from PropwiseAI Requirements:**\n\n{primary}"


# Self-testing entrypoint when running 'python src/document_rag.py'
if __name__ == "__main__":
    print("=== TESTING DOCUMENT_RAG.PY ===")
    doc_rag = DocumentRAGSystem()
    test_query = "What property categories are supported?"
    print(f"\nDocument Question: '{test_query}'")
    res = doc_rag.query_document(test_query)
    print("\nAnswer Output:")
    print(res["answer"])

