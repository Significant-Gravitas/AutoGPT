# Send Email

## What it is
The Send Email block is a tool for sending emails using SMTP (Simple Mail Transfer Protocol) credentials.

## What it does
This block allows users to send an email to a specified recipient with a custom subject and body. It uses provided SMTP credentials to connect to an email server and send the message.

## How it works
The block takes the recipient's email address, subject, and body of the email as inputs. It also requires SMTP credentials, including the server address, port, username, and password. The block then connects to the specified SMTP server, authenticates using the provided credentials, and sends the email. After attempting to send the email, it reports back whether the operation was successful or if an error occurred.

## Inputs
| Input | Description |
|-------|-------------|
| To Email | The email address of the recipient |
| Subject | The subject line of the email |
| Body | The main content of the email message |
| SMTP Credentials | Server, port, username, and password for authentication |

### SMTP Credentials Details
| Credential | Description | Default |
|------------|-------------|---------|
| SMTP Server | The address of the SMTP server | smtp.gmail.com |
| SMTP Port | The port number for the SMTP server | 25 |
| SMTP Username | The username for authenticating with the SMTP server | - |
| SMTP Password | The password for authenticating with the SMTP server | - |

## Outputs
| Output | Description |
|--------|-------------|
| Status | A message indicating whether the email was sent successfully |
| Error | If the email sending fails, this output provides details about the error that occurred |

## Possible use case
This block could be used in an automated customer support system. When a customer submits a support ticket through a website, the Send Email block could automatically send a confirmation email to the customer, acknowledging receipt of their request and providing them with a ticket number for future reference.