"""
PropwiseAI - Site Visit & Callback Form Handler Module (With Close Button)
Description:
    This dedicated module contains the UI form logic for processing:
    1. Customer Site Visit Requests (Preferred Date, Time Slot, Realtor Linking)
    2. Customer Callback Requests (Contact Info & Scheduled Status)
    
    Includes a top-right '✖ Close' button so customers can easily dismiss the form.
    Persists newly generated leads into data/leads.csv via LeadManager.
"""

import streamlit as st
from datetime import date
from typing import Dict, Any, Optional

# Import LeadManager from lead_manager.py
from src.lead_manager import LeadManager

def render_site_visit_form(property_data: Dict[str, Any], lead_manager: LeadManager) -> bool:
    """
    Renders an interactive Streamlit form to schedule a physical property site visit.
    Includes a '✖ Close' button to dismiss the modal form.
    
    Parameters:
        property_data (dict): Property details (title, property_id, broker info)
        lead_manager (LeadManager): Instance of LeadManager for CSV persistence
        
    Returns:
        bool: True if form was submitted or closed, False otherwise.
    """
    st.markdown("---")
    
    # Top Header Row with Title and Close Button
    h_col1, h_col2 = st.columns([6, 1])
    with h_col1:
        st.subheader(f"📅 Schedule Site Visit for: {property_data.get('title')}")
    with h_col2:
        if st.button("✖ Close", key=f"close_visit_{property_data.get('property_id')}", use_container_width=True):
            return True

    # Use Streamlit form container to group input fields together
    with st.form(key="site_visit_modal_form"):
        # Customer contact input fields
        customer_name = st.text_input("Customer Name", value="Ananya Sharma")
        phone_number = st.text_input("Mobile Number", value="+91 98765 43210")
        email_address = st.text_input("Email Address", value="ananya@example.com")

        # Split date and time slot inputs into two side-by-side columns
        col_date, col_time = st.columns(2)

        # Date picker input (restricts selection to today or future dates)
        visit_date = col_date.date_input("Preferred Date", min_value=date.today())

        # Selectbox dropdown for picking time slot
        time_slot = col_time.selectbox(
            "Preferred Time Slot",
            [
                "10:00 AM - 12:00 PM",
                "12:00 PM - 02:00 PM",
                "02:00 PM - 04:00 PM",
                "04:00 PM - 06:00 PM"
            ]
        )

        # Form submit button
        submit_button = st.form_submit_button("Confirm Site Visit Request")

        # Code executed when user clicks the submit button
        if submit_button:
            try:
                # Get assigned broker ID from property data dictionary
                broker_info = property_data.get('broker', {})
                assigned_broker_id = broker_info.get('broker_id')
                realtor_name = broker_info.get('name', 'Assigned Realtor')

                # Call LeadManager to create lead and save to data/leads.csv
                new_lead = lead_manager.create_lead(
                    customer_name=customer_name,
                    phone=phone_number,
                    email=email_address,
                    intent=property_data.get('listing_type', 'Buy'),
                    property_id=property_data.get('property_id'),
                    requested_action="site_visit",
                    preferred_date=str(visit_date),
                    preferred_time_slot=time_slot,
                    broker_id=assigned_broker_id
                )

                # Display green success confirmation badge & persistent toast notification
                if isinstance(new_lead, dict) and not new_lead.get('saved_to_main_csv', True):
                    st.warning("⚠️ Note: `data/leads.csv` is currently open in Excel or locked. Saved to temporary backup; changes will sync automatically when Excel closes.")
                
                lead_id = new_lead.get('lead_id', 'New') if isinstance(new_lead, dict) else 'New'
                prop_title = property_data.get('title', 'Selected Property')

                # Append Thank-You confirmation message directly to Chat History
                if "chat_history" in st.session_state:
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": (
                            f"🎉 **Thank You {customer_name}! Your Site Visit Request has been confirmed.**\n\n"
                            f"• 🆔 **Reference ID:** `{lead_id}`\n"
                            f"• 🏠 **Property:** {prop_title}\n"
                            f"• 📅 **Preferred Date:** {visit_date}\n"
                            f"• ⏰ **Time Slot:** {time_slot}\n"
                            f"• 👨‍💼 **Assigned Realtor:** {realtor_name}\n\n"
                            f"Our real estate representative will contact you shortly at **{phone_number}** to confirm!"
                        )
                    })

                toast_msg = f"🎉 Site Visit request registered successfully! Ref ID: {lead_id}."
                st.toast(toast_msg, icon="✅")
                return True
            except PermissionError:
                st.warning("⚠️ `data/leads.csv` is currently open in Microsoft Excel or another program. Please close Excel and try again.")
                return False
            except Exception as e:
                st.error(f"⚠️ Could not complete request: {e}")
                return False

    return False


def render_callback_form(property_data: Dict[str, Any], lead_manager: LeadManager) -> bool:
    """
    Renders an interactive Streamlit form to request an agent phone callback.
    Includes a '✖ Close' button to dismiss the modal form.
    
    Parameters:
        property_data (dict): Property details (title, property_id, broker info)
        lead_manager (LeadManager): Instance of LeadManager for CSV persistence
        
    Returns:
        bool: True if form was submitted or closed, False otherwise.
    """
    st.markdown("---")
    
    # Top Header Row with Title and Close Button
    h_col1, h_col2 = st.columns([6, 1])
    with h_col1:
        st.subheader(f"📞 Request Realtor Callback for: {property_data.get('title')}")
    with h_col2:
        if st.button("✖ Close", key=f"close_call_{property_data.get('property_id')}", use_container_width=True):
            return True

    # Streamlit form container
    with st.form(key="callback_modal_form"):
        # Customer contact input fields
        customer_name = st.text_input("Customer Name", value="Vikram Malhotra")
        phone_number = st.text_input("Mobile Number", value="+91 98123 99887")
        email_address = st.text_input("Email Address", value="vikram@example.com")

        # Form submit button
        submit_button = st.form_submit_button("Confirm Callback Request")

        # Code executed when user clicks the submit button
        if submit_button:
            try:
                # Get assigned broker ID from property data dictionary
                broker_info = property_data.get('broker', {})
                assigned_broker_id = broker_info.get('broker_id')
                realtor_name = broker_info.get('name', 'Assigned Realtor')

                # Call LeadManager to create lead and save to data/leads.csv
                new_lead = lead_manager.create_lead(
                    customer_name=customer_name,
                    phone=phone_number,
                    email=email_address,
                    intent="Callback",
                    property_id=property_data.get('property_id'),
                    requested_action="callback",
                    broker_id=assigned_broker_id
                )

                # Display persistent toast notification & append Thank You confirmation message to Chat History
                if isinstance(new_lead, dict) and not new_lead.get('saved_to_main_csv', True):
                    st.warning("⚠️ Note: `data/leads.csv` is currently open in Excel or locked. Saved to temporary backup; changes will sync automatically when Excel closes.")

                lead_id = new_lead.get('lead_id', 'New') if isinstance(new_lead, dict) else 'New'
                prop_title = property_data.get('title', 'Selected Property')

                # Append Thank-You confirmation message directly to Chat History
                if "chat_history" in st.session_state:
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": (
                            f"📞 **Thank You {customer_name}! Your Callback Request has been registered.**\n\n"
                            f"• 🆔 **Reference ID:** `{lead_id}`\n"
                            f"• 🏠 **Property:** {prop_title}\n"
                            f"• 📱 **Contact Number:** {phone_number}\n"
                            f"• 👨‍💼 **Assigned Realtor:** {realtor_name}\n\n"
                            f"Our agent **{realtor_name}** will call you back shortly to assist with your property inquiry!"
                        )
                    })

                toast_msg = f"📞 Callback request scheduled! Ref ID: {lead_id}."
                st.toast(toast_msg, icon="📞")
                return True
            except PermissionError:
                st.warning("⚠️ `data/leads.csv` is currently open in Microsoft Excel or another program. Please close Excel and try again.")
                return False
            except Exception as e:
                st.error(f"⚠️ Could not complete request: {e}")
                return False

    return False
