"""
PropwiseAI - System Configuration Module
Author: PropwiseAI Engineering Team
Description:
    Centralized configuration file defining workspace paths, CSV datasets,
    FAISS vector store persistence directories, embedding models, and default parameters.
"""

import os
from pathlib import Path

# Base workspace directory path
BASE_DIR = Path(__file__).resolve().parent.parent

# Data directory path containing raw datasets and requirements text
DATA_DIR = BASE_DIR / "data"
VECTOR_STORE_DIR = BASE_DIR / "vector_store"

# Ensure essential directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

# CSV dataset file paths
BROKERS_CSV_PATH = DATA_DIR / "brokers.csv"
PROPERTIES_CSV_PATH = DATA_DIR / "properties.csv"
LEADS_CSV_PATH = DATA_DIR / "leads.csv"
DOC_KNOWLEDGE_PATH = DATA_DIR / "document_knowledge.txt"

# FAISS Vector Index persistence directories
DOC_INDEX_PATH = VECTOR_STORE_DIR / "doc_faiss_index"
PROPERTY_INDEX_PATH = VECTOR_STORE_DIR / "property_faiss_index"

# Local Embedding Model configuration (Runs 100% offline without API key dependencies)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Supported property categories in PropwiseAI platform
SUPPORTED_CATEGORIES = [
    "Town House",
    "Modern Villa",
    "Apartment",
    "Single Family",
    "Office"
]

# Supported customer intent categories
SUPPORTED_INTENTS = {
    "buy": "Buy a property",
    "rent": "Rent a property",
    "investment": "Investment-oriented search",
    "explore": "Explore properties",
    "site_visit": "Schedule site visit",
    "callback": "Request agent callback",
    "enquiry": "Submit general property enquiry",
    "compare": "Compare selected properties",
    "doc_faq": "Platform requirements Q&A"
}

# Gemini LLM Integration Configuration (Optional: set GEMINI_API_KEY to enable live Gemini AI synthesis)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-flash")


