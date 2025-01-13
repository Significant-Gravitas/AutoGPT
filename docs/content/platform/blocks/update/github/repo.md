

# Gmail Integration Blocks Documentation

## Gmail Read Block

### What it is
A tool that allows you to read emails from your Gmail account.

### What it does
Retrieves and reads emails from your Gmail inbox based on specific search criteria.

### How it works
Connects to your Gmail account, searches for emails matching your query, and retrieves the email content including subject, body, and attachments.

### Inputs
- Credentials: Your Gmail account authentication credentials
- Query: Search terms to filter emails (e.g., "is:unread" for unread emails)
- Max Results: Maximum number of emails to retrieve (default: 10)

### Outputs
- Email: Individual email data including subject, sender, recipient, date, body, and attachments
- Emails: Complete list of retrieved emails
- Error: Any error message if something goes wrong

### Possible use case
Automatically monitoring incoming emails for specific keywords or checking unread messages from important contacts.

## Gmail Send Block

### What it is
A tool for sending emails through your Gmail account.

### What it does
Composes and sends emails to specified recipients using your Gmail account.

### How it works
Takes your email content and recipient information, formats it properly, and sends it through Gmail's system.

### Inputs
- Credentials: Your Gmail account authentication credentials
- To: Recipient's email address
- Subject: Email subject line
- Body: Main content of the email

### Outputs
- Result: Confirmation of email sending status
- Error: Any error message if something goes wrong

### Possible use case
Sending automated responses or notifications to clients or team members based on specific triggers.

## Gmail List Labels Block

### What it is
A tool for retrieving all labels (folders) from your Gmail account.

### What it does
Fetches and lists all labels/categories that exist in your Gmail account.

### How it works
Connects to Gmail and retrieves a complete list of your email labels and their IDs.

### Inputs
- Credentials: Your Gmail account authentication credentials

### Outputs
- Result: List of all Gmail labels with their IDs and names
- Error: Any error message if something goes wrong

### Possible use case
Creating an overview of email organization structure or preparing for email management automation.

## Gmail Add Label Block

### What it is
A tool for adding labels to specific emails in Gmail.

### What it does
Applies a specified label to a particular email message.

### How it works
Creates a new label if it doesn't exist, then applies it to the specified email.

### Inputs
- Credentials: Your Gmail account authentication credentials
- Message ID: Unique identifier of the target email
- Label Name: Name of the label to add

### Outputs
- Result: Confirmation of label addition
- Error: Any error message if something goes wrong

### Possible use case
Automatically categorizing incoming emails based on content or sender.

## Gmail Remove Label Block

### What it is
A tool for removing labels from specific emails in Gmail.

### What it does
Removes a specified label from a particular email message.

### How it works
Locates the specified label and removes it from the target email.

### Inputs
- Credentials: Your Gmail account authentication credentials
- Message ID: Unique identifier of the target email
- Label Name: Name of the label to remove

### Outputs
- Result: Confirmation of label removal
- Error: Any error message if something goes wrong

### Possible use case
Automatically updating email categories when certain conditions are met, such as when an task is completed.

