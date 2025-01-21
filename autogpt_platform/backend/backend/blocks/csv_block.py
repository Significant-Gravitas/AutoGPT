import os
from datetime import datetime
import csv
from io import StringIO
import base64
import json
from google.oauth2 import service_account
from googleapiclient.discovery import build

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import SchemaField

class CSVBlock(Block):
    scopes: list[str] = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]

    class Input(BlockSchema):
        csv_content: str = SchemaField(
            description="The content of the CSV file to write",
        )

        file_name: str = SchemaField(
        description="The name of the file to write",
        default = f"csv-{datetime.now().strftime('%Y%m%d-%H%M')}.csv"
        )

    class Output(BlockSchema):
        spreadsheet_link: str = SchemaField(
            description="The link to the created spreadsheet"
        )
        spreadsheet_id: str = SchemaField(
            description="The ID of the created spreadsheet"
        )

    def __init__(self):
        super().__init__(
            id="d61aad1d-7752-4e61-ac81-cc55e48c3a67",  # Replace this with a new unique ID
            description="This block writes data to a Google Sheets spreadsheet.",
            categories={BlockCategory.DATA},
            input_schema=CSVBlock.Input,
            output_schema=CSVBlock.Output,
            test_input={
                "csv_content": "name,age,city\nAlice,30,New York\nBob,25,Los Angeles",
                "file_name": "abc-123.csv",
            },
            test_output={
                "spreadsheet_link": "https://docs.google.com/spreadsheets/d/0f0bdc30-9606-4691-aa2d-807605b33b63",
                "spreadsheet_id": "95d29dd9-cb29-42fa-8632-058270978e21"
            }
        )
    
    def csv_excute(self, csv_content: str, file_name: str = None) -> dict:
        sa_base64 = os.getenv("GOOGLE_SPREADSHEET_SERVICE_ACCOUNT")
        if not sa_base64:
            raise Exception("GOOGLE_SPREADSHEET_SERVICE_ACCOUNT is not set")

        # Decode the base64 service account key
        sa_json = base64.b64decode(sa_base64).decode('utf-8')
        service_account_info = json.loads(sa_json)
        credentials = service_account.Credentials.from_service_account_info(
            service_account_info, 
            scopes=CSVBlock.scopes
        )

        # Initialize the Google Sheets and Drive APIs
        service = build("sheets", "v4", credentials=credentials)
        sheet = service.spreadsheets()
        drive_service = build("drive", "v3", credentials=credentials)

        # Set default filename if not provided
        if file_name is None:
            data_time = datetime.now().strftime("%Y%m%d-%H%M")
            file_name = f"csv-{data_time}.csv"

        # Create a new spreadsheet
        spreadsheet_body = {
            "properties": {"title": file_name}
        }

        spreadsheet = sheet.create(
            body=spreadsheet_body, 
            fields="spreadsheetId"
        ).execute()
        
        spreadsheet_id = spreadsheet.get("spreadsheetId")

        csv_file = StringIO(csv_content)

        # Use csv.reader to parse the CSV string
        csv_reader = csv.reader(csv_file)
        data = list(csv_reader)

        # Update the spreadsheet with parsed data
        body = {"values": data}

        # Update the first sheet of the new spreadsheet with the CSV data
        sheet.values().update(
            spreadsheetId=spreadsheet_id,
            range="Sheet1!A1",  # Starting from the first row and column
            valueInputOption="RAW",
            body=body
        ).execute()

        # Make the file public
        self.make_file_public(spreadsheet_id, drive_service)

        # Construct the spreadsheet link
        spreadsheet_link = f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}"

        return {
            "spreadsheet_link": spreadsheet_link,
            "spreadsheet_id": spreadsheet_id
        }
    
    def make_file_public(self, spreadsheet_id: str, drive_service):
        permission_body = {
            "type": "anyone",  # Make it accessible to anyone with the link
            "role": "reader"   # Use "writer" if you want others to edit the file
        }
        
        # Set the file permissions using the Drive API
        drive_service.permissions().create(
            fileId=spreadsheet_id,  # The ID of the file to make public
            body=permission_body
        ).execute()

    def run(self, input_data: Input, **kwargs) -> BlockOutput:
        try:
           # Extract the CSV content and file name from the input_data object
            csv_content = input_data.csv_content
            file_name = input_data.file_name  # This will use the default value if not provided
            if not csv_content:
                raise ValueError("CSV content is required")
            
            print("input =====================================================>", csv_content)

            # Call the csv_execute method and get the result
            result = self.csv_excute(csv_content, file_name)
            spreadsheet_link = result["spreadsheet_link"]
            spreadsheet_id = result["spreadsheet_id"]

            yield "spreadsheet_link", spreadsheet_link
            yield "spreadsheet_id", spreadsheet_id
        except Exception as e:
            yield "error", f"An error occurred: {e}"

