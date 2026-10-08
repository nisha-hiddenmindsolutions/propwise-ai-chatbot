"""
PropwiseAI - Customer AI Chatbot Web Application (PropertyFinder Sidebar Theme)
Author: PropwiseAI Team
Description:
    Clean, customer-facing Streamlit Web Application styled after PropwiseAI & PropertyFinder.
    Features:
      1. Sidebar with '+ New Search' primary button & 'Clear Chat' action.
      2. 'RECENT SEARCHES' history navigation list.
      3. PropwiseAI Branding Header
      4. Section 6 Welcome Flow & Quick Suggested Action Buttons
      5. LangGraph Stateful Conversational AI Agent
      6. Hybrid FAISS Property Result Cards with Realtor Contact & Recommendation Reasons
      7. Interactive Site-Visit & Callback Request Modal Forms (src/forms_handler.py)
"""

import sys
import os
from pathlib import Path
from datetime import date, datetime
import streamlit as st
import pandas as pd

# Fix project base path for module resolution
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

# Import system modules
from src.config import PROPERTIES_CSV_PATH, BROKERS_CSV_PATH
from src.graph import PropwiseAgentGraph
from src.lead_manager import LeadManager
from src.forms_handler import render_site_visit_form, render_callback_form

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="PropwiseAI - Real Estate AI Assistant",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# CUSTOM CSS STYLING (PROPERTYFINDER SIDEBAR + PROPWISEAI BRANDING)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Global font & background adjustments */
    .stApp {
        background-color: #F8FAFC;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E2E8F0;
    }

    /* + New Search Button Styling (Primary Pill Button) */
    .stSidebar div.stButton > button[key="btn_new_search"] {
        background: linear-gradient(135deg, #4F46E5 0%, #6366F1 100%) !important;
        color: #FFFFFF !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        border-radius: 14px !important;
        border: none !important;
        padding: 12px 24px !important;
        box-shadow: 0 4px 14px rgba(99, 102, 241, 0.35) !important;
        transition: all 0.2s ease !important;
    }
    .stSidebar div.stButton > button[key="btn_new_search"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.45) !important;
    }

    /* Clear Chat Button Styling */
    .stSidebar div.stButton > button[key="btn_clear_chat"] {
        background-color: #F1F5F9 !important;
        color: #475569 !important;
        font-weight: 600 !important;
        border-radius: 12px !important;
        border: 1px solid #E2E8F0 !important;
    }

    /* Recent Search Item Buttons */
    .stSidebar div.stButton > button[key^="recent_item_"] {
        background-color: #F8FAFC !important;
        color: #334155 !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 12px !important;
        text-align: left !important;
        font-size: 0.88rem !important;
        padding: 10px 14px !important;
        margin-bottom: 6px !important;
    }
    .stSidebar div.stButton > button[key^="recent_item_"]:hover {
        background-color: #EEF2FF !important;
        border-color: #C7D2FE !important;
        color: #4338CA !important;
    }
    
    /* Header Branding Banner */
    .propwise-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        padding: 24px 32px;
        border-radius: 16px;
        color: #FFFFFF;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.15);
    }
    .propwise-logo {
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #FFFFFF;
    }
    .propwise-logo span {
        color: #10B981;
    }
    .propwise-tagline {
        color: #94A3B8;
        font-size: 1rem;
        margin-top: 4px;
    }
    .status-badge {
        background-color: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid rgba(52, 211, 153, 0.3);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }

    /* Property Result Cards Styling */
    .property-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px -2px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .property-card:hover {
        box-shadow: 0 8px 24px -4px rgba(0, 0, 0, 0.08);
    }
    .badge-sale {
        background-color: #DCFCE7;
        color: #15803D;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.8rem;
        letter-spacing: 0.5px;
    }
    .badge-rent {
        background-color: #E0E7FF;
        color: #4338CA;
        padding: 4px 12px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.8rem;
        letter-spacing: 0.5px;
    }
    .price-tag {
        font-size: 1.6rem;
        font-weight: 800;
        color: #059669;
        margin-top: 8px;
    }
    .reason-box {
        background-color: #F0FDF4;
        border-left: 4px solid #10B981;
        padding: 10px 14px;
        border-radius: 6px;
        font-size: 0.9rem;
        color: #065F46;
        margin-top: 10px;
        margin-bottom: 14px;
        font-weight: 500;
    }
    .broker-box {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        padding: 12px 16px;
        border-radius: 10px;
        font-size: 0.88rem;
        color: #334155;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# INITIALIZE SESSION STATE OBJECTS
# -----------------------------------------------------------------------------
if "agent_graph" not in st.session_state:
    st.session_state.agent_graph = PropwiseAgentGraph()
if "lead_manager" not in st.session_state:
    st.session_state.lead_manager = LeadManager()
if "session_state" not in st.session_state:
    st.session_state.session_state = {
        "messages": [],
        "session_context": {},
        "matched_properties": [],
        "selected_property": None
    }
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {
            "role": "assistant",
            "content": "Hi! I can help you find a property that matches your requirements. Are you looking to buy, rent, invest, or explore properties?"
        }
    ]
if "recent_searches" not in st.session_state:
    st.session_state.recent_searches = [
        {"title": "Jaipur Luxury Search", "query": "3 BHK apartment in Jaipur under 70 lakh"},
        {"title": "Mumbai Rental Flat", "query": "2 BHK flat for rent in Mumbai up to 40k"},
        {"title": "Waterfront Villa Search", "query": "4 BHK Modern Villa with private garden"}
    ]

# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION & RECENT SEARCHES (PROPERTYFINDER SIDEBAR STYLE)
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="padding: 4px 0px 16px 0px;">
        <h2 style="font-weight: 800; font-size: 1.6rem; color: #4F46E5; margin: 0;">Propwise<span style="color: #10B981;">AI</span></h2>
    </div>
    """, unsafe_allow_html=True)

    # Primary Action: + New Search Button (Clears session & starts fresh search)
    if st.button("➕ New Search", key="btn_new_search", use_container_width=True):
        st.session_state.agent_graph = PropwiseAgentGraph()
        st.session_state.chat_history = [
            {
                "role": "assistant",
                "content": "Hi! I can help you find a property that matches your requirements. Are you looking to buy, rent, invest, or explore properties?"
            }
        ]
        st.session_state.session_state = {
            "messages": [],
            "session_context": {},
            "matched_properties": [],
            "selected_property": None
        }
        st.rerun()

    # Secondary Action: Clear Current Chat
    if st.button("🗑️ Clear Current Chat", key="btn_clear_chat", use_container_width=True):
        st.session_state.agent_graph = PropwiseAgentGraph()
        st.session_state.chat_history = [
            {
                "role": "assistant",
                "content": "Chat history cleared. What property are you looking for today?"
            }
        ]
        st.session_state.session_state["session_context"] = {}
        st.session_state.session_state["matched_properties"] = []
        st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style="font-size: 0.75rem; font-weight: 700; color: #64748B; letter-spacing: 1px; margin-bottom: 12px;">
        RECENT SEARCHES
    </div>
    """, unsafe_allow_html=True)

    # Render Recent Searches List
    for i, search_item in enumerate(st.session_state.recent_searches):
        s_title = search_item.get("title", "Property Search")
        s_query = search_item.get("query", s_title)
        
        if st.button(f"🔍 {s_title}", key=f"recent_item_{i}", use_container_width=True):
            st.session_state.quick_prompt = s_query
            st.rerun()

# -----------------------------------------------------------------------------
# HEADER BRANDING BANNER
# -----------------------------------------------------------------------------
st.markdown("""
<div class="propwise-header">
    <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
            <div class="propwise-logo">Propwise<span>AI</span></div>
            <div class="propwise-tagline">AI-Powered Real Estate Discovery & Customer Intelligence Platform</div>
        </div>
        <div>
            <span class="status-badge">🟢 Live AI Agent Active</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# AI CUSTOMER CHATBOT INTERFACE
# -----------------------------------------------------------------------------
st.subheader("🤖 PropwiseAI Property Search Agent")
st.caption("Ask in natural language (e.g. '3 BHK flat for buy in Jaipur under 70 lakh with parking')")

# SECTION 6: Welcome Flow Suggested Action Buttons
st.markdown("**Suggested actions:**")
col1, col2, col3, col4 = st.columns(4)
quick_prompt = None

if col1.button("🏠 Buy Property", use_container_width=True):
    quick_prompt = "I am looking to buy a property"
if col2.button("🔑 Rent Property", use_container_width=True):
    quick_prompt = "I am looking to rent a property"
if col3.button("💼 Investment", use_container_width=True):
    quick_prompt = "I am looking for an investment property"
if col4.button("🔍 Explore Properties", use_container_width=True):
    quick_prompt = "Show me available properties to explore"

st.write("")

# Render Chat History
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Render Property Cards if matches are present in message
        if "matched_properties" in msg and msg["matched_properties"]:
            st.markdown("---")
            for idx, prop in enumerate(msg["matched_properties"], 1):
                with st.container():
                    st.markdown('<div class="property-card">', unsafe_allow_html=True)
                    p_col1, p_col2 = st.columns([1, 2])
                    
                    with p_col1:
                        if prop.get("primary_image"):
                            st.image(prop["primary_image"], use_container_width=True)
                        badge_class = "badge-sale" if str(prop.get("listing_type")).lower() == "sale" else "badge-rent"
                        st.markdown(f'<span class="{badge_class}">{str(prop.get("listing_type")).upper()}</span>', unsafe_allow_html=True)
                        st.markdown(f'<div class="price-tag">{prop.get("price_formatted")}</div>', unsafe_allow_html=True)

                    with p_col2:
                        st.subheader(prop.get("title"))
                        st.write(f"📍 **Address:** {prop.get('address')} ({prop.get('city')})")
                        st.write(f"🛏️ **BHK:** {prop.get('bhk')} | 🚿 **Baths:** {prop.get('bathrooms')} | 📐 **Area:** {prop.get('area_sqft')} sq.ft.")
                        st.write(f"🛋️ **Furnishing:** {prop.get('furnishing')} | ✨ **Amenities:** {prop.get('amenities')}")
                        st.markdown(f'<div class="reason-box">💡 {prop.get("recommendation_reason")} (Match: {prop.get("match_score")}%)</div>', unsafe_allow_html=True)

                        broker = prop.get("broker", {})
                        if broker:
                            st.markdown(
                                f'<div class="broker-box">'
                                f'<b>Realtor:</b> {broker.get("name")} | 🏢 {broker.get("agency_name")}<br>'
                                f'📞 {broker.get("phone")} | 📜 RERA License: {broker.get("license")}'
                                f'</div>',
                                unsafe_allow_html=True
                            )

                        st.write("")
                        btn_c1, btn_c2 = st.columns(2)

                        with btn_c1:
                            if st.button("📅 Request Site Visit", key=f"visit_{msg.get('msg_id', 'm')}_{idx}"):
                                st.session_state.active_visit_prop = prop
                                st.session_state.show_visit_form = True

                        with btn_c2:
                            if st.button("📞 Request Callback", key=f"call_{msg.get('msg_id', 'm')}_{idx}"):
                                st.session_state.active_call_prop = prop
                                st.session_state.show_call_form = True

                    st.markdown('</div>', unsafe_allow_html=True)

# Render Site Visit Request Form Modal (src/forms_handler.py)
if st.session_state.get("show_visit_form") and st.session_state.get("active_visit_prop"):
    is_submitted = render_site_visit_form(
        property_data=st.session_state.active_visit_prop,
        lead_manager=st.session_state.lead_manager
    )
    if is_submitted:
        st.session_state.show_visit_form = False
        st.session_state.active_visit_prop = None
        st.rerun()

# Render Callback Request Form Modal (src/forms_handler.py)
if st.session_state.get("show_call_form") and st.session_state.get("active_call_prop"):
    is_submitted = render_callback_form(
        property_data=st.session_state.active_call_prop,
        lead_manager=st.session_state.lead_manager
    )
    if is_submitted:
        st.session_state.show_call_form = False
        st.session_state.active_call_prop = None
        st.rerun()

# Chat Input Bar
user_input = st.chat_input("Ask about properties, buy/rent options, locations, or specific property details...")
if quick_prompt:
    user_input = quick_prompt
elif st.session_state.get("quick_prompt"):
    user_input = st.session_state.get("quick_prompt")
    st.session_state.quick_prompt = None

if user_input:
    # Save search to Recent Searches list if unique
    new_title = user_input[:28] + "..." if len(user_input) > 28 else user_input
    if not any(item["query"] == user_input for item in st.session_state.recent_searches):
        st.session_state.recent_searches.insert(0, {"title": new_title, "query": user_input})
        st.session_state.recent_searches = st.session_state.recent_searches[:6]

    # Append user message
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    # Process message through LangGraph state machine agent
    with st.spinner("AI Chatbot is searching live inventory & processing context..."):
        result = st.session_state.agent_graph.process_message(
            user_input,
            st.session_state.session_state
        )

    # Update active context memory & matched properties
    if result.get("session_context"):
        st.session_state.session_state["session_context"] = result["session_context"]
    if result.get("matched_properties"):
        st.session_state.session_state["matched_properties"] = result["matched_properties"]

    msg_id = len(st.session_state.chat_history)
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": result.get("response_text", ""),
        "matched_properties": result.get("matched_properties", []),
        "msg_id": msg_id
    })
    st.rerun()
