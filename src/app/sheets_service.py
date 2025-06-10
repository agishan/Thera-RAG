import json
import streamlit as st
from datetime import datetime
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

class SheetsService:
    """Simple Google Sheets logging service"""
    
    def __init__(self, config):
        self.config = config
        self.service = self._init_service()
    
    def _init_service(self):
        """Initialize Google Sheets API service"""
        try:
            # Parse credentials
            if isinstance(self.config['google_sheets_creds_json'], str):
                creds_dict = json.loads(self.config['google_sheets_creds_json'])
            else:
                creds_dict = self.config['google_sheets_creds_json']
            
            # Create credentials and service
            credentials = service_account.Credentials.from_service_account_info(
                creds_dict,
                scopes=['https://www.googleapis.com/auth/spreadsheets']
            )
            
            return build('sheets', 'v4', credentials=credentials)
            
        except Exception as e:
            st.warning(f"Google Sheets not available: {e}")
            return None
    
    def setup_sheet(self):
        """Create and setup the logging sheet"""
        if not self.service:
            return False
        
        try:
            # Check if sheet exists
            sheet_metadata = self.service.spreadsheets().get(
                spreadsheetId=self.config['google_sheets_spreadsheet_id']
            ).execute()
            
            sheet_names = [s['properties']['title'] for s in sheet_metadata.get('sheets', [])]
            
            if self.config['sheets_name'] not in sheet_names:
                # Create sheet
                self.service.spreadsheets().batchUpdate(
                    spreadsheetId=self.config['google_sheets_spreadsheet_id'],
                    body={
                        'requests': [{
                            'addSheet': {
                                'properties': {'title': self.config['sheets_name']}
                            }
                        }]
                    }
                ).execute()
                
                # Add headers
                headers = [['Timestamp', 'Session ID', 'User Query', 'Assistant Response', 'Response Time (s)', 'Answer Length']]
                self.service.spreadsheets().values().update(
                    spreadsheetId=self.config['google_sheets_spreadsheet_id'],
                    range=f"{self.config['sheets_name']}!A1:F1",
                    valueInputOption='RAW',
                    body={'values': headers}
                ).execute()
            
            return True
            
        except Exception as e:
            st.error(f"Error setting up sheet: {e}")
            return False
    
    def log_interaction(self, session_id, query, answer, elapsed_time):
        """Log a chat interaction to Google Sheets"""
        if not self.service:
            return False
        
        try:
            row_data = [[
                str(datetime.now()),
                session_id,
                query,
                answer[:1000] if len(answer) > 1000 else answer,
                round(elapsed_time, 2),
                len(answer)
            ]]
            
            self.service.spreadsheets().values().append(
                spreadsheetId=self.config['google_sheets_spreadsheet_id'],
                range=f"{self.config['sheets_name']}!A:F",
                valueInputOption='RAW',
                insertDataOption='INSERT_ROWS',
                body={'values': row_data}
            ).execute()
            
            return True
            
        except Exception as e:
            st.error(f"Error logging to sheets: {e}")
            return False