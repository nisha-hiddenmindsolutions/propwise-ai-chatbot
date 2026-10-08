"""
PropwiseAI - CRM Lead & Site-Visit Manager (Step 5)
Description:
    This module handles creating and saving customer leads, site visit requests,
    and callback requests into data/leads.csv.
"""

import sys
import os
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd

# Ensure parent project root is in Python path for direct script execution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import LEADS_CSV_PATH, PROPERTIES_CSV_PATH, BROKERS_CSV_PATH

logger = logging.getLogger(__name__)

class LeadManager:
    """
    Manages customer leads, site visit bookings, and callback requests.
    Persists data directly into data/leads.csv.
    """

    def __init__(self):
        # Create empty DataFrame for storing leads
        self.leads_df: pd.DataFrame = pd.DataFrame()
        # Load existing leads from CSV file
        self.load_leads()

    def load_leads(self):
        """Reads data/leads.csv into pandas dataframe, merging backup CSV if present."""
        cols = [
            'lead_id', 'customer_name', 'phone', 'email', 'intent',
            'selected_property_id', 'requested_action', 'preferred_date',
            'preferred_time_slot', 'broker_id', 'source', 'status', 'created_at'
        ]
        
        # Initialize empty DataFrame with required headers if missing
        if not os.path.exists(LEADS_CSV_PATH):
            logger.warning(f"Leads CSV missing at {LEADS_CSV_PATH}. Initializing empty dataframe.")
            self.leads_df = pd.DataFrame(columns=cols)
        else:
            try:
                self.leads_df = pd.read_csv(LEADS_CSV_PATH)
                logger.info(f"Loaded {len(self.leads_df)} leads from CSV dataset.")
            except Exception as e:
                logger.error(f"Failed to read {LEADS_CSV_PATH}: {e}")
                self.leads_df = pd.DataFrame(columns=cols)

        # Check for backup CSV if leads were saved during file locking
        leads_dir = os.path.dirname(LEADS_CSV_PATH)
        alt_path = os.path.join(leads_dir, "leads_backup.csv")
        if os.path.exists(alt_path):
            try:
                backup_df = pd.read_csv(alt_path)
                if not backup_df.empty:
                    # Merge unique leads based on lead_id
                    combined = pd.concat([self.leads_df, backup_df], ignore_index=True)
                    self.leads_df = combined.drop_duplicates(subset=['lead_id'], keep='last')
                    logger.info(f"Merged backup leads. Total leads count: {len(self.leads_df)}")
            except Exception as backup_err:
                logger.error(f"Error loading backup CSV: {backup_err}")

    def save_leads(self) -> bool:
        """
        Safely saves self.leads_df to LEADS_CSV_PATH.
        Handles Windows PermissionError (e.g. when leads.csv is locked by Microsoft Excel) gracefully.
        
        Returns:
            bool: True if saved to main CSV file, False if fallback or error occurred.
        """
        leads_dir = os.path.dirname(LEADS_CSV_PATH)
        if leads_dir:
            os.makedirs(leads_dir, exist_ok=True)

        try:
            self.leads_df.to_csv(LEADS_CSV_PATH, index=False)
            logger.info(f"Saved {len(self.leads_df)} leads to {LEADS_CSV_PATH}")
            return True
        except (PermissionError, OSError) as e:
            logger.warning(f"Permission error writing to {LEADS_CSV_PATH}: {e}. Attempting fallback file.")
            alt_path = os.path.join(leads_dir, "leads_backup.csv")
            try:
                self.leads_df.to_csv(alt_path, index=False)
                logger.info(f"Saved backup leads to {alt_path}")
            except Exception as backup_err:
                logger.error(f"Error saving to backup leads file: {backup_err}")
            return False

    def create_lead(
        self,
        customer_name: str,
        phone: str,
        email: str,
        intent: str,
        property_id: Optional[str] = None,
        requested_action: str = "enquiry",
        preferred_date: str = "N/A",
        preferred_time_slot: str = "N/A",
        broker_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Creates a new customer lead record, assigns appropriate broker ID,
        appends to data/leads.csv, and re-saves the CSV file safely.
        """
        # Step 1: Auto-generate unique lead ID (e.g. lead_9004, lead_9005)
        next_id_number = 9001
        if not self.leads_df.empty and 'lead_id' in self.leads_df.columns:
            existing_ids = self.leads_df['lead_id'].astype(str).tolist()
            id_numbers = []
            for lid in existing_ids:
                if lid.startswith('lead_'):
                    try:
                        id_numbers.append(int(lid.replace('lead_', '')))
                    except ValueError:
                        pass
            if id_numbers:
                next_id_number = max(id_numbers) + 1

        new_lead_id = f"lead_{next_id_number}"

        # Step 2: Auto-lookup broker assigned to the selected property if broker_id is not given
        if not broker_id and property_id and os.path.exists(PROPERTIES_CSV_PATH):
            try:
                props_df = pd.read_csv(PROPERTIES_CSV_PATH)
                matched_prop = props_df[props_df['property_id'] == property_id]
                if not matched_prop.empty:
                    broker_id = str(matched_prop.iloc[0]['broker_id'])
            except Exception as e:
                logger.error(f"Error finding broker for property {property_id}: {e}")

        # Fallback to default broker if none assigned
        if not broker_id:
            broker_id = "brk_101"

        # Current UTC timestamp for lead creation
        created_timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

        # Map requested action to lead status string
        status_map = {
            "site_visit": "Pending Confirmation",
            "callback": "Callback Scheduled",
            "enquiry": "New Lead"
        }
        lead_status = status_map.get(requested_action, "New Lead")

        # Create dictionary for the new lead record
        new_lead_record = {
            'lead_id': new_lead_id,
            'customer_name': customer_name,
            'phone': phone,
            'email': email,
            'intent': intent.capitalize() if intent else 'Explore',
            'selected_property_id': property_id if property_id else 'N/A',
            'requested_action': requested_action,
            'preferred_date': preferred_date,
            'preferred_time_slot': preferred_time_slot,
            'broker_id': broker_id,
            'source': 'AI Chatbot',
            'status': lead_status,
            'created_at': created_timestamp
        }

        # Step 3: Append new row to DataFrame and safely re-save
        new_row_df = pd.DataFrame([new_lead_record])
        self.leads_df = pd.concat([self.leads_df, new_row_df], ignore_index=True)
        saved_ok = self.save_leads()
        new_lead_record['saved_to_main_csv'] = saved_ok
        logger.info(f"Created lead {new_lead_id} (Main CSV saved: {saved_ok})")

        return new_lead_record

    def get_all_leads(self) -> pd.DataFrame:
        """Returns all leads dataframe for CRM dashboard view."""
        self.load_leads()
        return self.leads_df

# Self-testing entrypoint when running 'python src/lead_manager.py'
if __name__ == "__main__":
    print("=== TESTING LEAD_MANAGER.PY ===")
    lm = LeadManager()
    print("\nExisting Total Leads:", len(lm.get_all_leads()))
    print("Creating sample test lead...")
    new_lead = lm.create_lead(
        customer_name="Test User",
        phone="9876543210",
        email="test@example.com",
        intent="Buy",
        requested_action="site_visit",
        preferred_date="2026-10-15",
        preferred_time_slot="Morning (10 AM - 1 PM)"
    )
    print("Created Lead Output:")
    print("  • Lead ID:", new_lead["lead_id"])
    print("  • Status:", new_lead["status"])

