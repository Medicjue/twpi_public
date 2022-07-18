from __future__ import print_function
import httplib2
import os

from apiclient import discovery
import os.path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

class ResultExporter:
    SECRET_PATH = '.credentials/client_secret.json'
    CREDS_PATH = '.credentials/cred.json'
    def __init__(self, conf_path):
        from configparser import ConfigParser
        self.creds = None
        self.scopes = ['https://www.googleapis.com/auth/spreadsheets']
        self.config = ConfigParser()
        self.config.read(conf_path)
        # The file client_secret.json stores the user's access and refresh tokens, and is
        # created automatically when the authorization flow completes for the first
        # time.
        if os.path.exists(self.CREDS_PATH):
            self.creds = Credentials.from_authorized_user_file(self.CREDS_PATH, self.scopes)
        # If there are no (valid) credentials available, let the user log in.
        if not self.creds or not self.creds.valid:
            if self.creds and self.creds.expired and self.creds.refresh_token:
                self.creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.SECRET_PATH, self.scopes)
                self.creds = flow.run_local_server(port=0)
            # Save the credentials for the next run
            with open(self.CREDS_PATH, 'w') as token:
                token.write(self.creds.to_json())
        self.google_spreadsheet_svc = build('sheets','v4', credentials=self.creds)


    def run(self):
        from mongo_dao import SentimentIndexDAO
        from datetime import datetime
        import pandas as pd

        output = []
        dao = SentimentIndexDAO()
        data = dao.query()
        
        for x in data:
            e = {}
            dt_str = x['dt']
            dt = datetime.strptime(dt_str, '%Y%m%d')
            e['dt'] = dt
            e['party'] = x['keyword']
            mdl_ver = x['mdl_ver']
            e['score'] = x['score']
            output.append(e)
        output_df = pd.DataFrame(output)
        output_df['dt'] = output_df['dt'].dt.strftime('%Y-%m-%d')

        spreadsheet_id = self.config['google']['spreadsheet_id']       
        sheet_name = self.config['google']['sheet_name']       

        self.google_spreadsheet_svc.spreadsheets().values().update(
                spreadsheetId=spreadsheet_id,
                range=sheet_name,
                valueInputOption='USER_ENTERED',
                body={
                    'majorDimension': 'ROWS',
                    'values': output_df.T.reset_index().T.values.tolist()
                },
            ).execute()


if __name__ == '__main__':
    result_exporter = ResultExporter(conf_path='conf/config.properties')
    result_exporter.run()