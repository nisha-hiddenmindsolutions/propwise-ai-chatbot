"""
PropwiseAI - Command Line Interface (CLI) Tester (Step 7)

Description:
    Terminal CLI application powered by LangGraph agent graph.
    Allows testing multi-turn property searches, document RAG Q&A, and property matching
    directly from the terminal.
"""

import sys
import os
from pathlib import Path

# Configure stdout for UTF-8 output on Windows terminal
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

# Set project base directory
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

# Import LangGraph agent graph
from src.graph import PropwiseAgentGraph

def run_cli():
    """Main CLI function for terminal interaction."""
    print("=" * 70)
    print("      PROPWISE AI - CUSTOMER CHATBOT CLI TESTER")
    print("=" * 70)
    print("Type your message below (or type 'exit' to quit).\n")

    # Initialize LangGraph Agent Graph
    agent = PropwiseAgentGraph()
    
    # Session state dictionary maintaining conversation memory
    session_state = {
        "messages": [],
        "session_context": {},
        "selected_property": None
    }

    # Initial AI greeting
    print("AI Chatbot: Hi! I can help you find a property that matches your requirements. Are you looking to buy, rent, invest, or explore properties?")

    while True:
        try:
            # Read user input from terminal
            user_input = input("\nYou: ").strip()
            
            # Skip empty inputs
            if not user_input:
                continue
                
            # Exit loop if user types exit or quit
            if user_input.lower() in ['exit', 'quit']:
                print("\nThank you for using PropwiseAI Chatbot CLI! Goodbye.")
                break

            # Process user message through LangGraph state machine
            result = agent.process_message(user_input, session_state)
            
            # Update session memory context
            if result.get("session_context"):
                session_state["session_context"] = result["session_context"]

            # Print AI response text
            print("\n" + "=" * 35 + " AI RESPONSE " + "=" * 35)
            print(result.get("response_text", ""))

            # Print matched property cards if available
            matched_props = result.get("matched_properties", [])
            if matched_props:
                print(f"\n[MATCHED PROPERTY CARDS ({len(matched_props)} Matches)]")
                for i, prop in enumerate(matched_props, 1):
                    print(f"\n  Card #{i}: {prop['title']}")
                    print(f"  • City: {prop['city']} ({prop['area']}) | Type: {prop['category']} ({prop['listing_type'].upper()})")
                    print(f"  • BHK: {prop['bhk']} Bedrooms | Price: {prop['price_formatted']} | Furnishing: {prop['furnishing']}")
                    print(f"  • Amenities: {prop['amenities']}")
                    print(f"  • Recommendation Rationale: {prop['recommendation_reason']} (Match: {prop['match_score']}%)")
                    print(f"  • Realtor: {prop['broker'].get('name')} | Phone: {prop['broker'].get('phone')} | Agency: {prop['broker'].get('agency_name')}")

            print("=" * 83)

        except KeyboardInterrupt:
            print("\nExiting CLI session...")
            break
        except Exception as e:
            print(f"\nError processing message: {e}")

if __name__ == "__main__":
    run_cli()
