
<file_name>autogpt_platform/backend/backend/blocks/hubspot/engagement.md</file_name>

## HubSpot Engagement Block

### What it is
A specialized block that manages HubSpot-related email communications and tracks user engagement metrics with those communications.

### What it does
This block performs two main functions:
1. Sends emails through HubSpot's system
2. Tracks and analyzes engagement metrics for contacts, including email opens, clicks, and replies

### How it works
The block connects to HubSpot's API and either sends emails or retrieves engagement data based on the selected operation. When tracking engagement, it calculates various metrics and produces an engagement score based on user interactions with emails.

### Inputs
- Credentials: HubSpot authentication credentials required to access the API
- Operation: The type of action to perform ("send_email" or "track_engagement")
- Email Data: A collection of information needed for sending emails, including:
  - Recipient email address
  - Email subject
  - Email content
- Contact ID: The unique identifier for a HubSpot contact when tracking engagement
- Timeframe Days: The number of days to look back when analyzing engagement metrics (default is 30 days)

### Outputs
- Result: The outcome of the operation, which varies depending on the selected operation:
  - For email sending: Contains the API response with email delivery information
  - For engagement tracking: Contains metrics such as:
    - Number of email opens
    - Number of email clicks
    - Number of email replies
    - Last engagement timestamp
    - Overall engagement score
- Status: Indicates the completion status of the operation ("email_sent" or "engagement_tracked")

### Possible use cases
1. Email Marketing Campaign:
   A marketing team could use this block to send promotional emails to customers and then track how well those emails perform by monitoring opens, clicks, and replies. The engagement score helps them understand which customers are most actively engaging with their communications.

2. Customer Success Analysis:
   A customer success team might use this block to monitor customer engagement over time, helping them identify which customers are highly engaged and which might need additional attention based on their engagement metrics.

3. Automated Follow-up System:
   An automated system could use this block to send initial emails to prospects and then track their engagement, triggering different follow-up actions based on how they interact with the emails.

