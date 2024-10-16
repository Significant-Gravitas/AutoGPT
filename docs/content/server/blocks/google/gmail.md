# Gmail

## Gmail Read

### What it is
A block that retrieves and reads emails from a Gmail account.

### What it does
This block searches for and retrieves emails from a specified Gmail account based on given search criteria. It can fetch multiple emails and provide detailed information about each email, including subject, sender, recipient, date, body content, and attachments.

### How it works
The block connects to the user's Gmail account using their credentials, performs a search based on the provided query, and retrieves the specified number of email messages. It then processes each email to extract relevant information and returns the results.

### Inputs
| Input | Description |
|-------|-------------|
| Credentials | The user's Gmail account credentials for authentication |
| Query | A search query to filter emails (e.g., "is:unread" for unread emails) |
| Max Results | The maximum number of emails to retrieve |

### Outputs
| Output | Description |
|--------|-------------|
| Email | Detailed information about a single email |
| Emails | A list of email data for multiple emails |
| Error | An error message if something goes wrong during the process |

### Possible use case
Automatically checking for new customer inquiries in a support email inbox and organizing them for quick response.

---

## Gmail Send

### What it is
A block that sends emails using a Gmail account.

### What it does
This block allows users to compose and send emails through their Gmail account. It handles the creation of the email message and sends it to the specified recipient.

### How it works
The block authenticates with the user's Gmail account, creates an email message with the provided details (recipient, subject, and body), and then sends the email using Gmail's API.

### Inputs
| Input | Description |
|-------|-------------|
| Credentials | The user's Gmail account credentials for authentication |
| To | The recipient's email address |
| Subject | The subject line of the email |
| Body | The main content of the email |

### Outputs
| Output | Description |
|--------|-------------|
| Result | Confirmation of the sent email, including a message ID |
| Error | An error message if something goes wrong during the process |

### Possible use case
Automatically sending confirmation emails to customers after they make a purchase on an e-commerce website.

---

## Gmail List Labels

### What it is
A block that retrieves all labels (categories) from a Gmail account.

### What it does
This block fetches and lists all the labels or categories that are set up in the user's Gmail account. These labels are used to organize and categorize emails.

### How it works
The block connects to the user's Gmail account and requests a list of all labels. It then processes this information and returns a simplified list of label names and their corresponding IDs.

### Inputs
| Input | Description |
|-------|-------------|
| Credentials | The user's Gmail account credentials for authentication |

### Outputs
| Output | Description |
|--------|-------------|
| Result | A list of labels, including their names and IDs |
| Error | An error message if something goes wrong during the process |

### Possible use case
Creating a dashboard that shows an overview of how many emails are in each category or label in a business email account.

---

## Gmail Add Label

### What it is
A block that adds a label to a specific email in a Gmail account.

### What it does
This block allows users to add a label (category) to a particular email message in their Gmail account. If the label doesn't exist, it creates a new one.

### How it works
The block first checks if the specified label exists in the user's Gmail account. If it doesn't, it creates the label. Then, it adds the label to the specified email message using the message ID.

### Inputs
| Input | Description |
|-------|-------------|
| Credentials | The user's Gmail account credentials for authentication |
| Message ID | The unique identifier of the email message to be labeled |
| Label Name | The name of the label to add to the email |

### Outputs
| Output | Description |
|--------|-------------|
| Result | Confirmation of the label addition, including the label ID |
| Error | An error message if something goes wrong during the process |

### Possible use case
Automatically categorizing incoming customer emails based on their content, adding labels like "Urgent," "Feedback," or "Invoice" for easier processing.

---

## Gmail Remove Label

### What it is
A block that removes a label from a specific email in a Gmail account.

### What it does
This block allows users to remove a label (category) from a particular email message in their Gmail account.

### How it works
The block first finds the ID of the specified label in the user's Gmail account. If the label exists, it removes it from the specified email message using the message ID.

### Inputs
| Input | Description |
|-------|-------------|
| Credentials | The user's Gmail account credentials for authentication |
| Message ID | The unique identifier of the email message to remove the label from |
| Label Name | The name of the label to remove from the email |

### Outputs
| Output | Description |
|--------|-------------|
| Result | Confirmation of the label removal, including the label ID |
| Error | An error message if something goes wrong during the process |

### Possible use case
Automatically removing the "Unread" label from emails after they have been processed by a customer service representative.