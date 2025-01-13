
<file_name>autogpt_platform/backend/backend/blocks/email_block.md</file_name>

## Send Email Block

### What it is
A communication block that enables sending emails through an SMTP (Simple Mail Transfer Protocol) server using provided credentials.

### What it does
This block sends emails to specified recipients with customizable subject lines and message content using configured email server credentials.

### How it works
The block connects to an email server using provided SMTP credentials, creates a formatted email message with the specified content, and sends it to the designated recipient. It uses secure connection protocols to ensure the email is sent safely and reliably.

### Inputs
- To Email: The recipient's email address where the message will be sent
- Subject: The title or subject line of the email
- Body: The main content or message of the email
- Credentials: Email server configuration including:
  - SMTP Server: The email server address (default: smtp.gmail.com)
  - SMTP Port: The connection port number (default: 25)
  - SMTP Username: Your email account username
  - SMTP Password: Your email account password or application-specific password

### Outputs
- Status: A message indicating whether the email was sent successfully
- Error: Information about what went wrong if the email sending failed

### Possible use case
- Sending automated notifications to team members when certain conditions are met
- Delivering system reports to administrators
- Sending welcome emails to new users
- Dispatching order confirmations in an e-commerce system
- Sending password reset links or security notifications
- Distributing newsletters or updates to subscribers

### Notes
- This block requires valid SMTP credentials to function
- It's recommended to use application-specific passwords when configuring Gmail or similar services
- The block uses TLS encryption for secure email transmission
- Default settings are configured for Gmail, but can be adjusted for other email providers

