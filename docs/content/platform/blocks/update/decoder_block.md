
## Send Email

### What it is
A communication block that enables sending emails through an SMTP server, designed to work with various email providers including Gmail.

### What it does
This block allows users to send emails by providing recipient details, email content, and SMTP server credentials. It handles the entire email sending process, from establishing a secure connection to delivering the message.

### How it works
The block establishes a secure connection with the specified SMTP server using provided credentials, constructs the email message with the given subject and body, and sends it to the designated recipient. After sending, it confirms the delivery status or reports any errors that occurred during the process.

### Inputs
- To Email: The recipient's email address where the message will be sent (e.g., recipient@example.com)
- Subject: The title or heading of the email that appears in the recipient's inbox
- Body: The main content of the email message
- Email Credentials:
  - SMTP Server: The address of the email server (default: smtp.gmail.com)
  - SMTP Port: The connection port number for the email server (default: 25)
  - SMTP Username: Your email address or account username (stored securely)
  - SMTP Password: Your email account password or app-specific password (stored securely)

### Outputs
- Status: A message indicating whether the email was sent successfully
- Error: Information about what went wrong if the email sending failed

### Possible use case
A company's automated notification system could use this block to:
- Send order confirmations to customers
- Deliver automated reports to team members
- Send password reset links to users
- Distribute newsletter updates to subscribers
- Send alert notifications when specific system events occur

The block is particularly useful in automated workflows where email communication needs to be triggered based on certain events or conditions.

