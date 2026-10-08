"""
PropwiseAI - Hybrid Property Retrieval Engine (Step 3)
Author: PropwiseAI Team
Description:
    This module combines Pandas CSV filtering (for exact filters like city, budget, BHK)
    with FAISS Vector Search (for matching descriptions & amenities like 'sea view', 'garden').
"""

import sys
import os
import logging
from typing import List, Dict, Any, Optional

# Ensure parent project root is in Python path for direct script execution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# Import config paths and local embedding manager from our previous steps
from src.config import (
    PROPERTIES_CSV_PATH,
    BROKERS_CSV_PATH,
    PROPERTY_INDEX_PATH
)
from src.embeddings import EmbeddingManager
from src.gemini_service import GeminiService

# Setup logger for printing helpful debug messages
logger = logging.getLogger(__name__)

class PropertyRAGSystem:
    """
    Hybrid search engine for properties.
    Combines pandas CSV filtering with FAISS vector search.
    """

    def __init__(self):
        # Load local embedding model singleton & Gemini Service
        self.embedding_manager = EmbeddingManager()
        self.gemini_service = GeminiService()
        
        # Create empty DataFrames for storing property inventory & brokers
        self.properties_df: pd.DataFrame = pd.DataFrame()

        self.brokers_df: pd.DataFrame = pd.DataFrame()
        self.vector_store: Optional[FAISS] = None

        # Load CSV files into pandas
        self.load_data()

        # Build or load FAISS vector index for properties
        self._initialize_or_load_vector_index()

    def load_data(self):
        """Reads properties.csv and brokers.csv using pandas."""
        # Check if CSV files exist before reading
        if not os.path.exists(PROPERTIES_CSV_PATH):
            raise FileNotFoundError(f"Missing properties CSV file at: {PROPERTIES_CSV_PATH}")
        if not os.path.exists(BROKERS_CSV_PATH):
            raise FileNotFoundError(f"Missing brokers CSV file at: {BROKERS_CSV_PATH}")

        # Read CSV data into dataframes
        self.properties_df = pd.read_csv(PROPERTIES_CSV_PATH)
        self.brokers_df = pd.read_csv(BROKERS_CSV_PATH)

        # Fill empty amenity or description strings so code doesn't crash on NaN
        self.properties_df['amenities'] = self.properties_df['amenities'].fillna('')
        self.properties_df['description'] = self.properties_df['description'].fillna('')
        
        logger.info(f"Successfully loaded {len(self.properties_df)} properties and {len(self.brokers_df)} brokers.")

    def _initialize_or_load_vector_index(self):
        """Checks if FAISS index is saved on disk. If yes, load it. If no, build it."""
        index_file = PROPERTY_INDEX_PATH / "index.faiss"
        
        # Try loading existing vector index to save CPU time
        if os.path.exists(index_file):
            try:
                self.vector_store = FAISS.load_local(
                    folder_path=str(PROPERTY_INDEX_PATH),
                    embeddings=self.embedding_manager.model,
                    allow_dangerous_deserialization=True
                )
                logger.info("Loaded property FAISS index from disk.")
                return
            except Exception as e:
                logger.warning(f"Could not load vector index: {e}. Rebuilding vector index now...")

        # If not present or loading failed, rebuild index
        self.rebuild_vector_index()

    def rebuild_vector_index(self):
        """Reads property details and creates FAISS vector embeddings for semantic search."""
        documents = []

        # Loop through each row in properties dataframe
        for idx, row in self.properties_df.iterrows():
            # Combine property features into one single text block for vector embedding
            text_representation = (
                f"Title: {row['title']}\n"
                f"Category: {row['category']}\n"
                f"Listing Type: {row['listing_type']}\n"
                f"City: {row['city']}\n"
                f"Area: {row['area']}\n"
                f"Price: {row['price_formatted']}\n"
                f"BHK: {row['bhk']} Bedrooms\n"
                f"Furnishing: {row['furnishing']}\n"
                f"Amenities: {row['amenities']}\n"
                f"Description: {row['description']}"
            )

            # Create a LangChain Document with metadata
            doc = Document(
                page_content=text_representation,
                metadata={
                    "property_id": str(row['property_id']),
                    "city": str(row['city']).lower(),
                    "listing_type": str(row['listing_type']).lower(),
                    "category": str(row['category']).lower(),
                    "bhk": int(row['bhk']) if pd.notnull(row['bhk']) else 0,
                    "price": float(row['price']) if pd.notnull(row['price']) else 0.0
                }
            )
            documents.append(doc)

        # Build FAISS vector database using local sentence-transformer embeddings
        self.vector_store = FAISS.from_documents(
            documents=documents,
            embedding=self.embedding_manager.model
        )

        # Save index locally so next restart is fast
        PROPERTY_INDEX_PATH.mkdir(parents=True, exist_ok=True)
        self.vector_store.save_local(folder_path=str(PROPERTY_INDEX_PATH))
        logger.info(f"Built and saved FAISS index for {len(documents)} properties.")

    def search_properties(
        self,
        intent: Optional[str] = None,
        city: Optional[str] = None,
        max_budget: Optional[float] = None,
        bhk: Optional[int] = None,
        category: Optional[str] = None,
        listing_type: Optional[str] = None,
        query_text: Optional[str] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search method.
        Step 1: Hard filter CSV using Pandas (city, budget, BHK, listing type).
        Step 2: Calculate FAISS vector similarity score for semantic text matching.
        Step 3: Rank results and attach broker details + recommendation reasons.
        """
        df = self.properties_df.copy()

        # --- STEP 1: PANDAS HARD FILTERING ---
        # Filter listing type (sale vs rent vs investment)
        if listing_type:
            df = df[df['listing_type'].str.lower() == listing_type.lower()]
        elif intent:
            if intent.lower() == 'buy':
                df = df[df['listing_type'].str.lower() == 'sale']
            elif intent.lower() == 'rent':
                df = df[df['listing_type'].str.lower() == 'rent']
            elif intent.lower() == 'investment':
                df = df[df['listing_type'].str.lower().isin(['sale', 'investment'])]

        # Filter by city (case insensitive search)
        if city:
            df = df[df['city'].str.lower().str.contains(city.lower(), na=False)]

        # Filter by maximum budget
        if max_budget and max_budget > 0:
            df = df[df['price'] <= max_budget]

        # Filter by minimum BHK required (only when searching residential properties)
        is_office_search = category and str(category).lower() in ['office', 'commercial']
        if bhk and bhk > 0 and not is_office_search:
            df = df[df['bhk'] >= bhk]

        # Filter by property category (e.g. Apartment, Villa, Office)
        if category:
            filtered_cat = df[df['category'].str.lower().str.contains(category.lower(), na=False)]
            # If specific category matches exist, use them; otherwise keep matching BHK/city properties
            if not filtered_cat.empty:
                df = filtered_cat

        # --- STEP 2: FAISS VECTOR SIMILARITY SCORES ---
        vector_scores = {}
        if query_text and self.vector_store:
            try:
                # Search top matches in FAISS vector store
                vector_results = self.vector_store.similarity_search_with_score(query_text, k=10)
                for doc, score in vector_results:
                    pid = doc.metadata.get("property_id")
                    # Convert distance score to 0.0 - 1.0 similarity scale
                    sim_score = max(0.0, 1.0 - (score / 2.0))
                    vector_scores[pid] = sim_score
            except Exception as e:
                logger.error(f"Error during vector similarity search: {e}")

        # --- STEP 3: MATCHING RATIONALE & BROKER MERGING ---
        results = []
        for idx, row in df.iterrows():
            pid = str(row['property_id'])

            # Look up broker details from brokers.csv using broker_id
            broker_data = {}
            matched_broker = self.brokers_df[self.brokers_df['broker_id'] == row['broker_id']]
            if not matched_broker.empty:
                b = matched_broker.iloc[0]
                broker_data = {
                    "broker_id": str(b['broker_id']),
                    "name": str(b['name']),
                    "agency_name": str(b['agency_name']),
                    "phone": str(b['phone']),
                    "email": str(b['email']),
                    "rating": float(b['rating']),
                    "license": str(b['license_number'])
                }

            # Generate human-readable recommendation reasons
            reasons = []
            if city and city.lower() in str(row['city']).lower():
                reasons.append(f"Located in {row['city']}")
            if bhk and row['bhk'] >= bhk:
                reasons.append(f"{row['bhk']} BHK match")
            if max_budget and row['price'] <= max_budget:
                reasons.append(f"Within budget ({row['price_formatted']})")
            if category and category.lower() in str(row['category']).lower():
                reasons.append(f"Property Category ({row['category']})")

            # Calculate match percentage
            base_score = 85.0
            if pid in vector_scores:
                base_score += (vector_scores[pid] * 14.0)
            
            final_match_score = min(99.0, round(base_score, 1))

            if reasons:
                recommendation_msg = "Recommended because it matches " + ", ".join(reasons) + "."
            else:
                recommendation_msg = f"Matches your active property search in {row['city']}."

            # Format primary image URL
            images_split = str(row['images']).split('|') if pd.notnull(row['images']) else []
            primary_img = images_split[0] if images_split else "https://images.unsplash.com/photo-1560518883-ce09059eeffa?w=800&q=80"

            # Create final dictionary for property card
            prop_item = row.to_dict()
            prop_item['broker'] = broker_data
            prop_item['match_score'] = final_match_score
            prop_item['recommendation_reason'] = recommendation_msg
            prop_item['primary_image'] = primary_img
            prop_item['image_list'] = images_split

            results.append(prop_item)

        # Sort properties by highest match score first
        results.sort(key=lambda x: x['match_score'], reverse=True)
        return results[:top_k]

    def get_property_by_id(self, property_id: str) -> Optional[Dict[str, Any]]:
        """Returns single property record by ID merged with broker details."""
        rows = self.properties_df[self.properties_df['property_id'] == property_id]
        if rows.empty:
            return None
        
        prop = rows.iloc[0].to_dict()
        b_rows = self.brokers_df[self.brokers_df['broker_id'] == prop['broker_id']]
        if not b_rows.empty:
            prop['broker'] = b_rows.iloc[0].to_dict()

        img_list = str(prop['images']).split('|') if pd.notnull(prop['images']) else []
        prop['primary_image'] = img_list[0] if img_list else ""
        prop['image_list'] = img_list
        return prop

    def find_similar_properties(self, property_id: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Finds similar properties using vector distance based on a selected property."""
        target = self.get_property_by_id(property_id)
        if not target:
            return []

        search_query = f"{target['category']} in {target['city']} {target['bhk']} BHK {target['description']}"
        matches = self.search_properties(
            city=target['city'],
            category=target['category'],
            query_text=search_query,
            top_k=limit + 1
        )

        # Remove the selected target property itself from recommendations
        similar = [p for p in matches if p['property_id'] != property_id]
        return similar[:limit]

    def compare_properties(self, property_ids: List[str]) -> Dict[str, Any]:
        """
        Section 16: Property Comparison.
        Returns side-by-side comparison data, tabular feature comparison, and factual analysis.
        """
        compared_props = []
        for pid in property_ids:
            p = self.get_property_by_id(pid)
            if p and p not in compared_props:
                compared_props.append(p)

        if not compared_props:
            return {
                "count": 0,
                "properties": [],
                "comparison_table": [],
                "summary": "No matching properties selected for comparison."
            }

        # Build comparison feature rows for tabular display
        table = []
        fields = [
            ("Title", "title"),
            ("City & Location", lambda x: f"{x['city']} ({x.get('area', '')})"),
            ("Listing Type", lambda x: str(x['listing_type']).upper()),
            ("Price / Rent", "price_formatted"),
            ("Property Category", "category"),
            ("Bedrooms (BHK)", lambda x: f"{x['bhk']} BHK"),
            ("Bathrooms", lambda x: f"{x.get('bathrooms', 'N/A')} Baths"),
            ("Size (Sq. Ft.)", lambda x: f"{x.get('sqft', x.get('area_sqft', 'N/A'))} sq.ft."),
            ("Furnishing", "furnishing"),
            ("Amenities", lambda x: str(x['amenities'])),
            ("Realtor Contact", lambda x: f"{x['broker'].get('name')} ({x['broker'].get('phone')})")
        ]

        for field_name, extractor in fields:
            row = {"Feature": field_name}
            for i, p in enumerate(compared_props, 1):
                col_name = f"Property #{i}: {p['title']}"
                if callable(extractor):
                    row[col_name] = extractor(p)
                else:
                    row[col_name] = str(p.get(extractor, "N/A"))
            table.append(row)

        # Try Gemini LLM intelligent synthesis if API key is active
        if len(compared_props) >= 2 and self.gemini_service.is_active:
            gemini_summary = self.gemini_service.compare_properties_with_gemini(compared_props[0], compared_props[1])
            if gemini_summary:
                return {
                    "count": len(compared_props),
                    "properties": compared_props,
                    "comparison_table": table,
                    "summary": f"🤖 **GEMINI AI COMPARISON REPORT:**\n\n{gemini_summary}"
                }

        # Offline fallback factual comparison summary
        summary_lines = []
        if len(compared_props) >= 2:
            p1 = compared_props[0]
            p2 = compared_props[1]
            summary_lines.append(f"⚖️ **PROPERTY COMPARISON REPORT:**\n")
            summary_lines.append(f"• **1. {p1['title']}** vs **2. {p2['title']}**")
            summary_lines.append(f"• **Price Comparison:** {p1['title']} is listed at **{p1['price_formatted']}**, while {p2['title']} is listed at **{p2['price_formatted']}**.")
            
            p1_sqft = p1.get('sqft', p1.get('area_sqft', 'N/A'))
            p2_sqft = p2.get('sqft', p2.get('area_sqft', 'N/A'))
            summary_lines.append(f"• **Space & BHK:** {p1['title']} has **{p1['bhk']} BHK ({p1_sqft} sq.ft.)**, whereas {p2['title']} has **{p2['bhk']} BHK ({p2_sqft} sq.ft.)**.")
            summary_lines.append(f"• **Furnishing & Amenities:** {p1['title']} is **{p1['furnishing']}** ({p1['amenities']}) vs {p2['title']} which is **{p2['furnishing']}** ({p2['amenities']}).")
        else:
            summary_lines.append(f"Selected 1 property: **{compared_props[0]['title']}**. Please select 2 properties to perform a side-by-side comparison.")

        return {
            "count": len(compared_props),
            "properties": compared_props,
            "comparison_table": table,
            "summary": "\n".join(summary_lines)
        }



# Self-testing entrypoint when running 'python src/property_rag.py'
if __name__ == "__main__":
    print("=== TESTING PROPERTY_RAG.PY ===")
    prop_rag = PropertyRAGSystem()
    print("\nSearching 3 BHK in Jaipur under ₹70 Lakh:")
    results = prop_rag.search_properties(city="Jaipur", bhk=3, max_budget=7000000)
    for p in results:
        print(f"  • {p['title']} | Price: {p['price_formatted']} | Match: {p['match_score']}%")

