"""
PropwiseAI - Intent Detection & Entity Extraction Engine (Enhanced Budget Parsing)
Description:
    Detects Intent (Buy, Rent, Invest, Explore, Site Visit, Callback, FAQ, Property Details Q&A)
    and extracts search attributes (City, BHK, Budget, Category, Amenities).
"""

import re
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

class NaturalLanguageExtractor:
    """
    Parses user text prompts to extract search parameters and Q&A intent.
    """

    def parse_user_prompt(self, text: str, active_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Reads user message and returns extracted fields.
        """
        clean_text = text.lower()
        extracted = {}

        # 1. Detect Intent (including specific Property Q&A)
        extracted['intent'] = self._detect_intent(clean_text, active_context)

        # 2. Extract City / Location
        extracted['city'] = self._extract_city(clean_text)

        # 3. Extract BHK Count
        extracted['bhk'] = self._extract_bhk(clean_text)

        # 4. Extract Maximum Budget
        extracted['max_budget'] = self._extract_budget(clean_text)

        # 5. Extract Property Category
        extracted['category'] = self._extract_category(clean_text)

        # 6. Extract Furnishing
        extracted['furnishing'] = self._extract_furnishing(clean_text)

        # 7. Extract Amenities
        extracted['amenities'] = self._extract_amenities(clean_text)

        # Merge with active context memory
        if active_context:
            clean_text = text.lower()
            new_explicit_search = any(kw in clean_text for kw in ['buy', 'rent', 'invest', 'want to', 'looking for', 'search'])
            
            for key in ['city', 'bhk', 'max_budget', 'category', 'furnishing']:
                # If category is Office or Commercial, reset BHK requirement
                if extracted.get('category') in ['Office', 'Commercial'] and key == 'bhk':
                    extracted['bhk'] = None
                elif extracted.get(key) is None and active_context.get(key) is not None:
                    # If this is a fresh search and category was not mentioned, don't force old sticky category
                    if new_explicit_search and key == 'category' and extracted.get('category') is None:
                        pass
                    else:
                        extracted[key] = active_context[key]

        return extracted

    def _detect_intent(self, text: str, context: Optional[Dict[str, Any]]) -> str:
        """Identifies action or question intent from prompt, enforcing domain boundaries."""
        
        # Out-of-domain & Non-Real-Estate Keyword Detection
        non_property_keywords = [
            'iphone', 'apple', 'phone', 'mobile', 'samsung', 'android', 'laptop', 'computer',
            'car', 'bike', 'pizza', 'food', 'burger', 'recipe', 'cook', 'cake',
            'joke', 'jokes', 'tell me a joke', 'tell me joke', 'funny', 'movie', 'song', 'music', 'weather', 'cricket', 'football',
            'who is', 'python', 'java', 'coding', 'program', 'clothes', 'shoes', 'dress', 'shoe'
        ]
        
        real_estate_domain_keywords = [
            'property', 'properties', 'flat', 'flats', 'apartment', 'apartments', 'villa', 'villas',
            'house', 'houses', 'home', 'homes', 'office', 'offices', 'commercial', 'townhouse',
            'bhk', 'bedroom', 'bedrooms', 'room', 'rooms', 'bath', 'bathroom', 'bathrooms',
            'sqft', 'square feet', 'carpet area', 'rent', 'rental', 'lease', 'buy', 'sale', 'purchase',
            'invest', 'investment', 'site visit', 'callback', 'compare', 'realtor', 'broker',
            'price', 'cost', 'budget', 'lakh', 'crore', 'cr', 'lac', 'thousand', 'k',
            'jaipur', 'mumbai', 'delhi', 'gurgaon', 'noida', 'bangalore', 'pune', 'hyderabad',
            'furnishing', 'furnished', 'amenities', 'parking', 'balcony', 'lift', 'gym', 'pool',
            'security', 'propwise', 'explore', 'looking for', 'search', 'hi', 'hello', 'hey'
        ]

        has_non_property = any(kw in text for kw in non_property_keywords)
        # Use word boundary matching so single letters like 'k' or 'l' inside 'joke' don't trigger false matches
        has_real_estate = any(re.search(r'\b' + re.escape(kw) + r'\b', text) for kw in real_estate_domain_keywords)

        if has_non_property and not has_real_estate:
            return 'out_of_domain'

        words = text.split()
        if not has_real_estate and len(words) > 0:
            return 'out_of_domain'

        # Check for site visit keywords
        if any(word in text for word in ['site visit', 'view property', 'schedule visit', 'book visit']):
            return 'site_visit'
            
        # Check for callback keywords
        if any(word in text for word in ['callback', 'call me', 'contact agent', 'phone call']):
            return 'callback'

        # Property specific detail Q&A keywords (Section 14 & Section 22)
        qna_keywords = [
            'is it', 'is this', 'does it', 'has it', 'can i', 'which floor', 'when can i',
            'bathroom', 'bathrooms', 'bath', 'baths',
            'how many bathroom', 'how many bathrooms',
            'how many bedroom', 'how many bedrooms', 'how many room',
            'what is the price', 'how much', 'cost', 'rent rate',
            'size', 'sqft', 'square feet', 'area of', 'carpet area',
            'parking', 'garage', 'furnished', 'furnishing',
            'possession', 'ready to move', 'immediate',
            'amenities', 'tell me about', 'details of this', 'in this flat', 'in this apartment', 'in this house', 'in this property'
        ]
        
        # If text is a question about a property attribute, mark as property_qna
        if any(kw in text for kw in qna_keywords) and not any(search_kw in text for search_kw in ['buy a', 'rent a', 'looking for a', 'search for a']):
            return 'property_qna'

        # Check for similar property keywords (Section 15)
        if any(word in text for word in ['similar', 'like this', 'resembles', 'other options like', 'comparable']):
            return 'similar'

        # Check for comparison keywords
        if any(word in text for word in ['compare', 'difference between', 'vs']):
            return 'compare'

            
        # Check for rent keywords
        if any(word in text for word in ['rent', 'rental', 'lease', 'monthly']) or any(w in text for w in ['for rent', 'to rent']):
            return 'rent'
            
        # Check for buy keywords
        if any(word in text for word in ['buy', 'purchase', 'sale', 'buying']) or any(w in text for w in ['to buy', 'for buy']):
            return 'buy'
            
        # Check for investment keywords
        if any(word in text for word in ['invest', 'investment', 'roi', 'yield']):
            return 'investment'
            
        # Check for platform FAQ Q&A keywords
        if any(word in text for word in ['what is propwise', 'how does propwise', 'rule', 'requirement', 'policy', 'category', 'categories', 'supported categories', 'platform']):
            return 'doc_faq'

        # Fallback intent (do not stick property_qna or out_of_domain to future turns)
        if context and context.get('intent') and context.get('intent') not in ['property_qna', 'out_of_domain']:
            return context['intent']
        return 'explore'

    def _extract_city(self, text: str) -> Optional[str]:
        known_cities = ['jaipur', 'mumbai', 'delhi', 'gurgaon', 'noida', 'bangalore', 'pune', 'hyderabad']
        for city in known_cities:
            if city in text:
                return city.capitalize()
        return None

    def _extract_bhk(self, text: str) -> Optional[int]:
        match = re.search(r'(\d+)\s*(bhk|bedroom|bed|room)', text)
        if match:
            return int(match.group(1))
        return None

    def _extract_budget(self, text: str) -> Optional[float]:
        """
        Parses financial amounts in Lakhs, Crores, or Thousands (including comma formatting).
        """
        # Match Crores (Cr)
        cr_match = re.search(r'(\d+(?:\.\d+)?)\s*(cr|crore|crores)', text)
        if cr_match:
            return float(cr_match.group(1)) * 10000000.0

        # Match Lakhs (L)
        lakh_match = re.search(r'(\d+(?:\.\d+)?)\s*(lakh|lakhs|lac|lacs|l)', text)
        if lakh_match:
            return float(lakh_match.group(1)) * 100000.0

        # Match Thousands (k / thousand)
        k_match = re.search(r'(\d+(?:\.\d+)?)\s*(k|thousand|thousands)', text)
        if k_match:
            return float(k_match.group(1)) * 1000.0

        # Match numbers with commas or direct digits (e.g., '40,000' or '40000')
        comma_num_match = re.search(r'(?:under|upto|max|budget of|around|to|for)\s*₹?\s*(\d{1,3}(?:,\d{3})+|\d{4,8})', text)
        if comma_num_match:
            num_str = comma_num_match.group(1).replace(',', '')
            return float(num_str)

        return None

    def _extract_category(self, text: str) -> Optional[str]:
        category_map = {
            'apartment': 'Apartment',
            'flat': 'Apartment',
            'villa': 'Modern Villa',
            'town house': 'Town House',
            'townhouse': 'Town House',
            'single family': 'Single Family',
            'house': 'Single Family',
            'office': 'Office',
            'commercial': 'Office'
        }
        for keyword, category_name in category_map.items():
            if keyword in text:
                return category_name
        return None

    def _extract_furnishing(self, text: str) -> Optional[str]:
        if 'fully furnished' in text or 'fully-furnished' in text:
            return 'Fully-Furnished'
        if 'semi furnished' in text or 'semi-furnished' in text:
            return 'Semi-Furnished'
        if 'furnished' in text:
            return 'Furnished'
        if 'unfurnished' in text:
            return 'Unfurnished'
        return None

    def _extract_amenities(self, text: str) -> List[str]:
        amenity_keywords = {
            'parking': 'Parking',
            'garage': 'Parking',
            'garden': 'Garden',
            'lawn': 'Garden',
            'balcony': 'Balcony',
            'lift': 'Lift',
            'elevator': 'Lift',
            'gym': 'Gym',
            'pool': 'Swimming Pool',
            'security': 'Security'
        }
        found_amenities = []
        for keyword, amenity_name in amenity_keywords.items():
            if keyword in text and amenity_name not in found_amenities:
                found_amenities.append(amenity_name)
        return found_amenities

# Self-testing entrypoint when running 'python src/extractor.py'
if __name__ == "__main__":
    print("=== TESTING EXTRACTOR.PY ===")
    extractor = NaturalLanguageExtractor()
    test_prompt = "I want to buy a 3 BHK apartment in Jaipur under 70 lakh with parking"
    print(f"\nUser Input: '{test_prompt}'")
    extracted_data = extractor.parse_user_prompt(test_prompt)
    print("\nExtracted Attributes:")
    for key, value in extracted_data.items():
        print(f"  • {key}: {value}")

