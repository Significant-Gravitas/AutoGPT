import unittest
import os
import sys
import win32com.client

def filter_emails_by_conversation(messages):

    messages.Sort("[ReceivedTime]", True) # Sort by received time, newest first

    # Create a dictionary to store messages by conversation topic
    conversations = {}

    # Group messages by conversation topic
    for message in messages:
        topic = message.ConversationTopic
        if topic not in conversations:
            conversations[topic] = [message]
        else:
            conversations[topic].append(message)

    # Filter out earlier messages in each conversation
    filtered_messages = []
    for messages in conversations.values():
        latest_message = messages[0]
        for message in messages:
            if message.SentOn > latest_message.SentOn:
                latest_message = message
        filtered_messages.append(latest_message)

    # Print the filtered messages
    if filtered_messages:
        print(f"Filtered {len(messages)} messages, keeping the latest in each conversation:")
    else:
        print("No messages to filter.")
    return filtered_messages

def search_sent_emails(query):
    """Search emails and return the results"""
    # Connect to Outlook
    outlook = win32com.client.Dispatch("Outlook.Application")
    namespace = outlook.GetNamespace("MAPI")
    # Choose Sent Items folder
    sent_items = namespace.GetDefaultFolder(5)

    # Define search term
    keyword = query
    search_query = f"@SQL=\"urn:schemas:mailheader:subject\" LIKE '%{keyword}%' OR \"urn:schemas:mailheader:body\" LIKE '%{keyword}%' OR \"urn:schemas:mailheader:from\" LIKE '%{keyword}%' OR \"urn:schemas:mailheader:cc\" LIKE '%{keyword}%' OR \"http://schemas.microsoft.com/mapi/proptag/0x0037001f\" LIKE '%{keyword}%'"
    # search_query = "[Body] = '{search_query}'"

    # Perform the search and get the emails
    emails = sent_items.Items.Restrict(search_query)

    filter_emails_by_conversation(emails)

    # Print found messages
    contents = "";
    if emails:
        for email in emails:
            if email:
                contents += (f"<Email>Subject: {email.Subject} Sent: ({email.SentOn})\r\n{email.Body})</Email>")
        return contents;
    else:
        return (f"No emails found with '{query}'")
    