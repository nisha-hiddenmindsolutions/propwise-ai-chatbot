"""
PropwiseAI - Grounded Gemini LLM Integration Engine
Author: PropwiseAI Team
Description:
    Provides Gemini AI model synthesis strictly grounded in properties.csv
    and document_knowledge.txt records. Eliminates hallucinations.
"""

import sys
import os
import logging
from typing import List, Dict, Any, Optional

# Ensure parent project root is in Python path for direct script execution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

from src.config import GEMINI_API_KEY, GEMINI_MODEL_NAME

logger = logging.getLogger(__name__)

class GeminiService:
    """
    Wrapper for Google Gemini AI model.
    Synthesizes natural language comparisons and RAG answers strictly grounded in CSV & document records.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", "")
        self.is_active = False
        self.model = None

        if GENAI_AVAILABLE and self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel("gemini-1.5-flash")
                self.is_active = True
                logger.info("Gemini Grounded LLM model successfully initialized!")
            except Exception as e:
                logger.warning(f"Could not initialize Gemini LLM: {e}")

    def compare_properties_with_gemini(self, property_1: Dict[str, Any], property_2: Dict[str, Any]) -> str:
        """
        Uses Gemini LLM to synthesize a factual comparison report strictly grounded in properties.csv data.
        """
        if not self.is_active or not self.model:
            return ""

        prompt = f"""
You are the senior PropwiseAI Real Estate Advisor.
Compare the following two real estate properties factually using ONLY the provided CSV database fields.

PROPERTY 1 (ID: {property_1.get('property_id')}):
- Title: {property_1.get('title')}
- City & Area: {property_1.get('city')} ({property_1.get('area')})
- Listing Type: {property_1.get('listing_type')}
- Price: {property_1.get('price_formatted')}
- Category: {property_1.get('category')}
- BHK: {property_1.get('bhk')} BHK
- Size: {property_1.get('area_sqft', property_1.get('sqft'))} sq.ft.
- Furnishing: {property_1.get('furnishing')}
- Amenities: {property_1.get('amenities')}
- Description: {property_1.get('description')}

PROPERTY 2 (ID: {property_2.get('property_id')}):
- Title: {property_2.get('title')}
- City & Area: {property_2.get('city')} ({property_2.get('area')})
- Listing Type: {property_2.get('listing_type')}
- Price: {property_2.get('price_formatted')}
- Category: {property_2.get('category')}
- BHK: {property_2.get('bhk')} BHK
- Size: {property_2.get('area_sqft', property_2.get('sqft'))} sq.ft.
- Furnishing: {property_2.get('furnishing')}
- Amenities: {property_2.get('amenities')}
- Description: {property_2.get('description')}

CRITICAL GROUNDING INSTRUCTIONS:
1. Base your comparison STRICTLY and EXCLUSIVELY on the property data fields provided above.
2. Do NOT invent prices, future appreciation rates, investment returns, or unlisted amenities.
3. Present factual differences first.

Write a concise comparison in clean markdown covering:
1. ⚖️ **Price & Value Comparison**
2. 📐 **Space & BHK Breakdown**
3. 🛋️ **Furnishing & Amenities Difference**
4. 💡 **Factual Recommendation** (Which profile best suits each property based on actual CSV data).
"""
        try:
            response = self.model.generate_content(prompt)
            if response and hasattr(response, 'text') and response.text:
                return response.text.strip()
            return ""
        except Exception as e:
            logger.error(f"Gemini comparison error: {e}")
            return ""

    def synthesize_rag_response_with_gemini(self, query: str, retrieved_chunks: List[str]) -> str:
        """
        Synthesizes a document Q&A response strictly grounded in retrieved FAISS passages from document_knowledge.txt.
        """
        if not self.is_active or not self.model or not retrieved_chunks:
            return ""

        context_text = "\n\n".join([f"Passage {i+1}:\n{c}" for i, c in enumerate(retrieved_chunks)])

        prompt = f"""
You are the official PropwiseAI Assistant.
Answer the user's question strictly using ONLY the retrieved passages from the PropwiseAI Functional Requirements document below.

USER QUESTION: {query}

RETRIEVED DOCUMENT PASSAGES:
{context_text}

CRITICAL RULES:
1. Answer using ONLY the facts stated in the passages above.
2. Do NOT invent policies, features, or rules outside the provided context.
3. If the answer is not contained in the passages, state clearly: "This information is not covered in the PropwiseAI documentation."
4. If the question is unrelated to real estate, properties, or PropwiseAI, decline by stating: "I am PropwiseAI, a dedicated real estate search assistant. I can only help you search, compare, and book real estate properties (apartments, flats, villas, offices, townhouses) across supported cities."

Answer:
"""
        try:
            response = self.model.generate_content(prompt)
            if response and hasattr(response, 'text') and response.text:
                return response.text.strip()
            return ""
        except Exception as e:
            logger.error(f"Gemini RAG synthesis error: {e}")
            return ""

# Self-testing entrypoint when running 'python src/gemini_service.py'
if __name__ == "__main__":
    print("=== TESTING GEMINI_SERVICE.PY ===")
    service = GeminiService()
    if service.is_active:
        print("Gemini API Key detected & Active!")
    else:
        print("Offline Fallback Mode: GEMINI_API_KEY is not set (Runs 100% offline gracefully).")
