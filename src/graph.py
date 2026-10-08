"""
PropwiseAI - LangGraph Conversation Workflow Orchestrator (No Cards on Missing Info)
Description:
    Stateful conversational workflow. Does not render property cards when mandatory
    search details (like location) are missing and follow-up questions are being asked.
"""

import sys
import os
import logging
from typing import Dict, Any, List, Optional, TypedDict

# Ensure parent project root is in Python path for direct script execution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langgraph.graph import StateGraph, END

# Import modules from previous steps
from src.extractor import NaturalLanguageExtractor
from src.property_rag import PropertyRAGSystem
from src.document_rag import DocumentRAGSystem
from src.lead_manager import LeadManager

logger = logging.getLogger(__name__)

# State dictionary schema
class PropwiseState(TypedDict):
    user_input: str
    messages: List[Dict[str, str]]
    intent: str
    session_context: Dict[str, Any]
    matched_properties: List[Dict[str, Any]]
    selected_property: Optional[Dict[str, Any]]
    missing_info: List[str]
    doc_answer: Optional[str]
    qna_answer: Optional[str]
    lead_created: Optional[Dict[str, Any]]
    response_text: str

class PropwiseAgentGraph:
    """
    Main LangGraph agent pipeline connecting all modules together.
    """

    def __init__(self):
        self.extractor = NaturalLanguageExtractor()
        self.property_rag = PropertyRAGSystem()
        self.document_rag = DocumentRAGSystem()
        self.lead_manager = LeadManager()
        self.graph = self._build_graph()

    def _build_graph(self):
        """Constructs the LangGraph state machine graph."""
        workflow = StateGraph(PropwiseState)

        workflow.add_node("understand_and_extract", self.node_understand_and_extract)
        workflow.add_node("evaluate_missing_info", self.node_evaluate_missing_info)
        workflow.add_node("property_search", self.node_property_search)
        workflow.add_node("document_rag_qna", self.node_document_rag_qna)
        workflow.add_node("property_specific_qna", self.node_property_specific_qna)
        workflow.add_node("compare_properties", self.node_compare_properties)
        workflow.add_node("similar_properties", self.node_similar_properties)
        workflow.add_node("generate_response", self.node_generate_response)

        workflow.set_entry_point("understand_and_extract")

        # Dynamic routing
        workflow.add_conditional_edges(
            "understand_and_extract",
            self.route_by_intent,
            {
                "out_of_domain": "generate_response",
                "doc_faq": "document_rag_qna",
                "property_qna": "property_specific_qna",
                "compare": "compare_properties",
                "similar": "similar_properties",
                "property_flow": "evaluate_missing_info"
            }
        )

        workflow.add_edge("evaluate_missing_info", "property_search")
        workflow.add_edge("property_search", "generate_response")
        workflow.add_edge("document_rag_qna", "generate_response")
        workflow.add_edge("property_specific_qna", "generate_response")
        workflow.add_edge("compare_properties", "generate_response")
        workflow.add_edge("similar_properties", "generate_response")
        workflow.add_edge("generate_response", END)

        return workflow.compile()

    def route_by_intent(self, state: PropwiseState) -> str:
        """Routes execution based on intent."""
        intent = state.get("intent", "explore")
        if intent == "out_of_domain":
            return "out_of_domain"
        if intent == "doc_faq":
            return "doc_faq"
        if intent == "property_qna":
            return "property_qna"
        if intent == "compare":
            return "compare"
        if intent == "similar":
            return "similar"
        return "property_flow"



    def node_understand_and_extract(self, state: PropwiseState) -> Dict[str, Any]:
        """Node 1: Extracts details and updates context memory."""
        user_text = state.get("user_input", "")
        existing_context = state.get("session_context", {})

        extracted = self.extractor.parse_user_prompt(user_text, active_context=existing_context)

        clean_prompt = user_text.lower()
        new_explicit_search = any(kw in clean_prompt for kw in ['buy', 'rent', 'invest', 'looking for', 'want to', 'search for'])

        updated_context = dict(existing_context)
        
        # If user initiates a fresh search, clear previous category/bhk if not in new prompt
        if new_explicit_search:
            if extracted.get("category"):
                updated_context["category"] = extracted["category"]
            elif "category" in updated_context and not any(cat in clean_prompt for cat in ['office', 'commercial', 'flat', 'apartment', 'villa', 'house']):
                updated_context.pop("category", None)

        for k, v in extracted.items():
            if v is not None:
                updated_context[k] = v

        # For Office/Commercial, ensure BHK requirement is cleared
        if updated_context.get("category") in ["Office", "Commercial"]:
            updated_context["bhk"] = None

        return {
            "intent": extracted.get("intent", "explore"),
            "session_context": updated_context
        }

    def node_evaluate_missing_info(self, state: PropwiseState) -> Dict[str, Any]:
        """
        Node 2: Dynamic Adaptive Follow-Up Logic.
        Checks missing parameters dynamically (Location -> BHK -> Budget).
        """
        ctx = state.get("session_context", {})
        intent = ctx.get("intent", "buy")
        category = str(ctx.get("category", "")).lower()
        missing_fields = []

        # Check fields dynamically (skip BHK for Office / Commercial properties)
        if not ctx.get("city"):
            missing_fields.append("preferred location (e.g. Jaipur, Mumbai, Delhi)")
        elif not ctx.get("bhk") and intent in ["buy", "rent"] and category not in ["office", "commercial"]:
            missing_fields.append("BHK / bedroom requirement")

        return {"missing_info": missing_fields}

    def node_property_search(self, state: PropwiseState) -> Dict[str, Any]:
        """
        Node 3: Invokes Step 3 Hybrid Property Search ONLY if mandatory info is present.
        If location is missing, returns matched_properties = [] so no cards render yet!
        """
        ctx = state.get("session_context", {})
        missing = state.get("missing_info", [])
        user_query = state.get("user_input", "")

        # If mandatory info like location is missing, do NOT search or render property cards yet!
        if missing:
            return {"matched_properties": []}

        # Otherwise execute hybrid search
        matches = self.property_rag.search_properties(
            intent=ctx.get("intent"),
            city=ctx.get("city"),
            max_budget=ctx.get("max_budget"),
            bhk=ctx.get("bhk"),
            category=ctx.get("category"),
            query_text=user_query,
            top_k=4
        )

        return {"matched_properties": matches}

    def node_document_rag_qna(self, state: PropwiseState) -> Dict[str, Any]:
        """Node 4: Invokes Document Vector RAG."""
        query = state.get("user_input", "")
        rag_result = self.document_rag.query_document(query)
        return {"doc_answer": rag_result["answer"], "matched_properties": []}

    def node_property_specific_qna(self, state: PropwiseState) -> Dict[str, Any]:
        """Node 5: Answers specific property data Q&A (Section 14)."""
        user_query = state.get("user_input", "").lower()
        matched = state.get("matched_properties", [])
        
        if not matched:
            ctx = state.get("session_context", {})
            matched = self.property_rag.search_properties(
                city=ctx.get("city"),
                bhk=ctx.get("bhk"),
                top_k=1
            )

        if not matched:
            return {
                "qna_answer": "I don't have property details loaded in context. Please search for a property first!",
                "matched_properties": []
            }

        target_prop = matched[0]
        prop_title = target_prop.get("title", "Selected Property")

        if any(w in user_query for w in ['furnished', 'furnishing', 'furniture']):
            furn = target_prop.get("furnishing", "N/A")
            return {
                "qna_answer": f"Yes, **{prop_title}** is **{furn}**.",
                "matched_properties": []
            }

        if any(w in user_query for w in ['size', 'sqft', 'square feet', 'square foot', 'area', 'carpet area']):
            area = target_prop.get("area_sqft", "N/A")
            return {
                "qna_answer": f"The total carpet area for **{prop_title}** is **{area} sq.ft.**",
                "matched_properties": []
            }

        if any(w in user_query for w in ['bathroom', 'bathrooms', 'bath', 'baths']):
            baths = target_prop.get("bathrooms", "N/A")
            return {
                "qna_answer": f"The **{prop_title}** features **{baths} bathrooms**.",
                "matched_properties": []
            }

        if any(w in user_query for w in ['bedroom', 'bedrooms', 'bhk', 'rooms']):
            bhk = target_prop.get("bhk", "N/A")
            return {
                "qna_answer": f"The **{prop_title}** has **{bhk} BHK / Bedrooms**.",
                "matched_properties": []
            }

        if 'floor' in user_query:
            flr = target_prop.get("floor", "N/A")
            return {
                "qna_answer": f"**{prop_title}** is located on the **{flr}**.",
                "matched_properties": []
            }

        if any(w in user_query for w in ['possession', 'move in', 'ready', 'timeline']):
            timeline = target_prop.get("possession_timeline", "N/A")
            return {
                "qna_answer": f"The possession timeline for **{prop_title}** is **{timeline}**.",
                "matched_properties": []
            }

        if 'parking' in user_query or 'garage' in user_query:
            amenities = target_prop.get("amenities", "")
            if 'Parking' in amenities:
                ans = f"Yes, dedicated parking is included with **{prop_title}**."
            else:
                ans = f"Parking is not explicitly listed for **{prop_title}**."
            return {
                "qna_answer": ans,
                "matched_properties": []
            }

        if any(w in user_query for w in ['price', 'cost', 'rent', 'rate', 'budget', 'pricing', 'how much cost', 'how much price']) or ('how much' in user_query and not any(w in user_query for w in ['sqft', 'area', 'size'])):
            price = target_prop.get("price_formatted", "N/A")
            return {
                "qna_answer": f"The price for **{prop_title}** is **{price}**.",
                "matched_properties": []
            }

        desc = target_prop.get("description", "")
        return {
            "qna_answer": f"**{prop_title}** ({target_prop.get('price_formatted')}):\n{desc}",
            "matched_properties": []
        }

    def node_compare_properties(self, state: PropwiseState) -> Dict[str, Any]:
        """Node for Section 16 Property Comparison."""
        matched = state.get("matched_properties", [])
        if not matched:
            ctx = state.get("session_context", {})
            matched = self.property_rag.search_properties(
                city=ctx.get("city"),
                bhk=ctx.get("bhk"),
                top_k=2
            )

        if len(matched) < 2:
            return {
                "qna_answer": "Please perform a search with at least 2 properties first so we can compare them side-by-side!",
                "matched_properties": matched
            }

        prop_ids = [p["property_id"] for p in matched[:2]]
        comp_res = self.property_rag.compare_properties(prop_ids)

        return {
            "qna_answer": comp_res["summary"],
            "matched_properties": comp_res["properties"]
        }

    def node_similar_properties(self, state: PropwiseState) -> Dict[str, Any]:
        """Node for Section 15 Similar Property Search."""
        matched = state.get("matched_properties", [])
        ctx = state.get("session_context", {})

        similar_list = []
        if matched:
            target_id = matched[0]["property_id"]
            similar_list = self.property_rag.find_similar_properties(target_id, limit=3)

        if not similar_list:
            similar_list = self.property_rag.search_properties(
                city=ctx.get("city"),
                category=ctx.get("category"),
                max_budget=ctx.get("max_budget"),
                bhk=ctx.get("bhk"),
                top_k=3
            )

        if similar_list:
            loc = ctx.get("city", "your search area")
            resp_text = f"Here are similar property recommendations in **{loc}** matching your preferences:"
        else:
            resp_text = "I couldn't find exact similar property matches in our database."

        return {
            "qna_answer": resp_text,
            "matched_properties": similar_list
        }

    def node_generate_response(self, state: PropwiseState) -> Dict[str, Any]:
        """Node 6: Synthesizes final response text."""
        intent = state.get("intent")
        user_input = state.get("user_input", "")

        # Out-of-Domain Guardrail Response (No Property Cards)
        if intent == "out_of_domain":
            return {
                "matched_properties": [],
                "response_text": "I am PropwiseAI, a dedicated real estate search assistant. I can only help you search, compare, and book real estate properties (apartments, flats, villas, offices, townhouses) across supported cities."
            }

        qna_ans = state.get("qna_answer")
        if qna_ans:
            return {"response_text": qna_ans}

        doc_ans = state.get("doc_answer")
        if doc_ans:
            return {"response_text": doc_ans}

        ctx = state.get("session_context", {})
        missing = state.get("missing_info", [])
        matches = state.get("matched_properties", [])

        intent = ctx.get("intent", "explore")
        city = ctx.get("city")
        bhk = ctx.get("bhk")
        category = ctx.get("category", "apartment")
        budget = ctx.get("max_budget")

        req_parts = []
        if bhk: req_parts.append(f"{bhk} BHK")
        req_parts.append(category.lower() if category != "property" else "apartment")
        if intent in ["buy", "rent"]:
            req_parts.append(f"for {intent}")
        if city:
            req_parts.append(f"in {city}")

        summary_text = " ".join(req_parts)

        budget_text = ""
        if budget:
            if budget >= 10000000:
                budget_text = f", with a budget up to ₹{budget/10000000:.2f} Cr."
            elif budget >= 100000:
                budget_text = f", with a budget up to ₹{budget/100000:.0f} Lakh."
            else:
                budget_text = f", with a budget up to ₹{budget:,.0f}/month."

        confirmation_heading = f"Got it. You’re looking for a {summary_text}{budget_text}"

        response_lines = [confirmation_heading]

        if missing:
            response_lines.append(f"\nCould you please specify your {missing[0]} so I can narrow down the search?")
        elif matches:
            response_lines.append("\nHere are the closest matching properties:")
        else:
            response_lines.append("\nNo exact matching properties were found for these exact constraints. Try broadening your budget or location!")

        return {"response_text": "\n".join(response_lines)}

    def process_message(self, user_text: str, session_state: Dict[str, Any]) -> Dict[str, Any]:
        """Public method to run graph flow."""
        initial_state: PropwiseState = {
            "user_input": user_text,
            "messages": session_state.get("messages", []),
            "intent": session_state.get("session_context", {}).get("intent", "explore"),
            "session_context": session_state.get("session_context", {}),
            "matched_properties": session_state.get("matched_properties", []),
            "selected_property": session_state.get("selected_property"),
            "missing_info": [],
            "doc_answer": None,
            "qna_answer": None,
            "lead_created": None,
            "response_text": ""
        }

        final_state = self.graph.invoke(initial_state)
        return final_state

# Self-testing entrypoint when running 'python src/graph.py'
if __name__ == "__main__":
    print("=== TESTING GRAPH.PY (LANGGRAPH STATE MACHINE) ===")
    agent = PropwiseAgentGraph()
    state = {"messages": [], "session_context": {}, "selected_property": None}
    test_msg = "Looking for a 2 BHK flat for rent in Mumbai"
    print(f"\nUser Input: '{test_msg}'")
    out = agent.process_message(test_msg, state)
    print("\nAI Response Text:")
    print(out.get("response_text"))
    print("\nMatched Property Cards Count:", len(out.get("matched_properties", [])))

