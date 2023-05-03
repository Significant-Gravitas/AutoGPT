import os
from unittest.mock import patch
from email.message import EmailMessage
from email_plugin import (
    send_email,
    read_emails,
    imap_open,
    send_email_with_attachment_internal,
    bothEmailAndPwdSet,
    adjust_imap_folder_for_gmail,
    enclose_with_quotes,
    split_imap_search_command,
)
from unittest.mock import mock_open
import unittest
from functools import partial

MOCK_FROM = "sender@example.com"
MOCK_PWD = "secret"
MOCK_TO = "test@example.com"
MOCK_DATE = "Fri, 21 Apr 2023 10:00:00 -0000"
MOCK_CONTENT = "Test message\n"
MOCK_SUBJECT = "Test Subject"
MOCK_IMAP_SERVER = "imap.example.com"
MOCK_SMTP_SERVER = "smtp.example.com"
MOCK_SMTP_PORT = "587"

MOCK_DRAFT_FOLDER = "Example/Drafts"
MOCK_ATTACHMENT_PATH = "example/file.txt"
MOCK_ATTACHMENT_NAME = "file.txt"


class TestEmailPlugin(unittest.TestCase):
    @patch.dict(
        os.environ,
        {
            "EMAIL_ADDRESS": "test@example.com",
            "EMAIL_PASSWORD": "test_password",
        },
    )
    def test_both_email_and_pwd_set(self):
        self.assertTrue(bothEmailAndPwdSet())

    @patch.dict(
        os.environ,
        {
            "EMAIL_PASSWORD": "test_password",
        },
        clear=True,
    )
    def test_email_not_set(self):
        self.assertFalse(bothEmailAndPwdSet())

    @patch.dict(
        os.environ,
        {
            "EMAIL_ADDRESS": "",
            "EMAIL_PASSWORD": "test_password",
        },
        clear=True,
    )
    def test_email_not_set_2(self):
        self.assertFalse(bothEmailAndPwdSet())

    @patch.dict(
        os.environ,
        {
            "EMAIL_ADDRESS": "test@example.com",
        },
        clear=True,
    )
    def test_pwd_not_set(self):
        self.assertFalse(bothEmailAndPwdSet())

    @patch.dict(os.environ, {}, clear=True)
    def test_both_email_and_pwd_not_set(self):
        self.assertFalse(bothEmailAndPwdSet())

    def test_adjust_imap_folder_for_gmail_normal_cases(self):
        self.assertEqual(
            adjust_imap_folder_for_gmail("Sent", "user@gmail.com"),
            '"[Gmail]/Sent Mail"',
        )
        self.assertEqual(
            adjust_imap_folder_for_gmail("Drafts", "user@googlemail.com"),
            "[Gmail]/Drafts",
        )
        self.assertEqual(
            adjust_imap_folder_for_gmail("Inbox", "user@gmail.com"), "Inbox"
        )

    def test_adjust_imap_folder_for_gmail_case_insensitivity(self):
        self.assertEqual(
            adjust_imap_folder_for_gmail("SeNT", "user@GMail.com"),
            '"[Gmail]/Sent Mail"',
        )
        self.assertEqual(
            adjust_imap_folder_for_gmail("DRAFTS", "user@gOogLemail.com"),
            "[Gmail]/Drafts",
        )
        self.assertEqual(
            adjust_imap_folder_for_gmail("InbOx", "user@gmail.com"), "InbOx"
        )

    def test_adjust_imap_folder_for_gmail_non_gmail_sender(self):
        self.assertEqual(adjust_imap_folder_for_gmail("Sent", "user@yahoo.com"), "Sent")
        self.assertEqual(
            adjust_imap_folder_for_gmail("Drafts", "user@hotmail.com"), "Drafts"
        )
        self.assertEqual(
            adjust_imap_folder_for_gmail("SENT", "gmail@hotmail.com"), "SENT"
        )

    def test_adjust_imap_folder_for_gmail_edge_cases(self):
        self.assertEqual(adjust_imap_folder_for_gmail("", "user@gmail.com"), "")
        self.assertEqual(adjust_imap_folder_for_gmail("Inbox", ""), "Inbox")
        self.assertEqual(adjust_imap_folder_for_gmail("", ""), "")

    def test_enclose_with_quotes(self):
        assert enclose_with_quotes("REVERSE DATE") == '"REVERSE DATE"'
        assert enclose_with_quotes('"My Search"') == '"My Search"'
        assert enclose_with_quotes("'test me'") == "'test me'"
        assert enclose_with_quotes("ALL") == "ALL"
        assert enclose_with_quotes("quotes needed") == '"quotes needed"'
        assert enclose_with_quotes("   whitespace  ") == '"   whitespace  "'
        assert enclose_with_quotes("whitespace\te") == '"whitespace\te"'
        assert enclose_with_quotes("\"mixed quotes'") == "\"mixed quotes'"
        assert enclose_with_quotes("'mixed quotes\"") == "'mixed quotes\""

    def test_split_imap_search_command(self):
        self.assertEqual(split_imap_search_command("SEARCH"), ["SEARCH"])
        self.assertEqual(
            split_imap_search_command("SEARCH UNSEEN"), ["SEARCH", "UNSEEN"]
        )
        self.assertEqual(
            split_imap_search_command("  SEARCH   UNSEEN  "), ["SEARCH", "UNSEEN"]
        )
        self.assertEqual(
            split_imap_search_command(
                "FROM speixoto@caicm.ca SINCE 01-JAN-2022 BEFORE 01-FEB-2023 HAS attachment xls OR HAS attachment xlsx"
            ),
            [
                "FROM",
                "speixoto@caicm.ca SINCE 01-JAN-2022 BEFORE 01-FEB-2023 HAS attachment xls OR HAS attachment xlsx",
            ],
        )
        self.assertEqual(
            split_imap_search_command("BODY here is my long body"),
            ["BODY", "here is my long body"],
        )
        self.assertEqual(split_imap_search_command(""), [])

    @patch("imaplib.IMAP4_SSL")
    @patch.dict(
        os.environ,
        {
            "EMAIL_ADDRESS": MOCK_FROM,
            "EMAIL_PASSWORD": MOCK_PWD,
            "EMAIL_IMAP_SERVER": MOCK_IMAP_SERVER,
        },
    )
    def test_imap_open(self, mock_imap):
        # Test imapOpen function
        imap_folder = "inbox"
        imap_open(imap_folder, MOCK_FROM, MOCK_PWD)

        # Check if the IMAP object was created and used correctly
        mock_imap.assert_called_once_with(MOCK_IMAP_SERVER)
        mock_imap.return_value.login.assert_called_once_with(MOCK_FROM, MOCK_PWD)
        mock_imap.return_value.select.assert_called_once_with(imap_folder)

    # Test for successful email sending without attachment
    @patch("smtplib.SMTP", autospec=True)
    @patch.dict(
        os.environ,
        {
            "EMAIL_ADDRESS": MOCK_FROM,
            "EMAIL_PASSWORD": MOCK_PWD,
            "EMAIL_SMTP_HOST": MOCK_SMTP_SERVER,
            "EMAIL_SMTP_PORT": MOCK_SMTP_PORT,
        },
    )
    def test_send_email_no_attachment(self, mock_smtp):
        result = send_email(MOCK_TO, MOCK_SUBJECT, MOCK_CONTENT)
        assert result == f"Email was sent to {MOCK_TO}!"

        mock_smtp.assert_called_once_with(MOCK_SMTP_SERVER, MOCK_SMTP_PORT)

        # Check if the SMTP object was created and used correctly
        context = mock_smtp.return_value.__enter__.return_value
        context.ehlo.assert_called()
        context.starttls.assert_called_once()
        context.login.assert_called_once_with(MOCK_FROM, MOCK_PWD)
        context.send_message.assert_called_once()
        context.quit.assert_called_once()

    # Test for reading emails in a specific folder with a specific search command
    @patch("imaplib.IMAP4_SSL")
    @patch.dict(
        os.environ,
        {
            "EMAIL_ADDRESS": MOCK_FROM,
            "EMAIL_PASSWORD": MOCK_PWD,
            "EMAIL_IMAP_SERVER": MOCK_IMAP_SERVER,
        },
    )
    def test_read_emails(self, mock_imap):
        assert os.getenv("EMAIL_ADDRESS") == MOCK_FROM

        # Create a mock email message
        message = EmailMessage()
        message["From"] = MOCK_FROM
        message["To"] = MOCK_TO
        message["Date"] = MOCK_DATE
        message["Subject"] = MOCK_SUBJECT
        message.set_content(MOCK_CONTENT)

        # Set up mock IMAP server behavior
        mock_imap.return_value.search.return_value = (None, [b"1"])
        mock_imap.return_value.fetch.return_value = (None, [(b"1", message.as_bytes())])

        # Test read_emails function
        result = read_emails("inbox", "UNSEEN")
        expected_result = [
            {
                "From": MOCK_FROM,
                "To": MOCK_TO,
                "Date": MOCK_DATE,
                "CC": "",
                "Subject": MOCK_SUBJECT,
                "Message Body": MOCK_CONTENT,
            }
        ]
        assert result == expected_result

        # Check if the IMAP object was created and used correctly
        mock_imap.return_value.login.assert_called_once_with(MOCK_FROM, MOCK_PWD)
        mock_imap.return_value.select.assert_called_once_with("inbox")
        mock_imap.return_value.search.assert_called_once_with(None, "UNSEEN")
        mock_imap.return_value.fetch.assert_called_once_with(b"1", "(BODY.PEEK[])")

    # Test for reading empty emails
    @patch("imaplib.IMAP4_SSL")
    @patch.dict(
        os.environ,
        {
            "EMAIL_ADDRESS": MOCK_FROM,
            "EMAIL_PASSWORD": MOCK_PWD,
            "EMAIL_IMAP_SERVER": MOCK_IMAP_SERVER,
        },
    )
    def test_read_empty_emails(self, mock_imap):
        assert os.getenv("EMAIL_ADDRESS") == MOCK_FROM

        # Set up mock IMAP server behavior
        mock_imap.return_value.search.return_value = (None, [b"0"])
        mock_imap.return_value.fetch.return_value = (None, [])

        # Test read_emails function
        result = read_emails("inbox", "UNSEEN")
        expected = "There are no Emails in your folder `inbox` "
        expected += "when searching with imap command `UNSEEN`"
        assert result == expected

        # Check if the IMAP object was created and used correctly
        mock_imap.return_value.login.assert_called_once_with(MOCK_FROM, MOCK_PWD)
        mock_imap.return_value.select.assert_called_once_with("inbox")
        mock_imap.return_value.search.assert_called_once_with(None, "UNSEEN")
        mock_imap.return_value.fetch.assert_called_once_with(b"0", "(BODY.PEEK[])")

    # Test for reading emails in a specific folder
    # with a specific search command with EMAIL_MARK_AS_SEEN=True
    @patch("imaplib.IMAP4_SSL")
    @patch.dict(
        os.environ,
        {
            "EMAIL_ADDRESS": MOCK_FROM,
            "EMAIL_PASSWORD": MOCK_PWD,
            "EMAIL_IMAP_SERVER": MOCK_IMAP_SERVER,
            "EMAIL_MARK_AS_SEEN": "True",
        },
    )
    def test_read_emails_mark_as_read_true(self, mock_imap):
        assert os.getenv("EMAIL_ADDRESS") == MOCK_FROM

        # Create a mock email message
        message = EmailMessage()
        message["From"] = MOCK_FROM
        message["To"] = MOCK_TO
        message["Date"] = MOCK_DATE
        message["Subject"] = MOCK_SUBJECT
        message.set_content(MOCK_CONTENT)

        # Set up mock IMAP server behavior
        mock_imap.return_value.search.return_value = (None, [b"1"])
        mock_imap.return_value.fetch.return_value = (None, [(b"1", message.as_bytes())])

        # Test read_emails function
        result = read_emails("inbox", "UNSEEN")
        expected_result = [
            {
                "From": MOCK_FROM,
                "To": MOCK_TO,
                "Date": MOCK_DATE,
                "CC": "",
                "Subject": MOCK_SUBJECT,
                "Message Body": MOCK_CONTENT,
            }
        ]
        assert result == expected_result

        # Check if the IMAP object was created and used correctly
        mock_imap.return_value.login.assert_called_once_with(MOCK_FROM, MOCK_PWD)
        mock_imap.return_value.select.assert_called_once_with("inbox")
        mock_imap.return_value.search.assert_called_once_with(None, "UNSEEN")
        mock_imap.return_value.fetch.assert_called_once_with(b"1", "(RFC822)")

    # Test for reading emails in a specific folder
    # with a specific search command with EMAIL_MARK_AS_SEEN=False
    @patch("imaplib.IMAP4_SSL")
    @patch.dict(
        os.environ,
        {
            "EMAIL_ADDRESS": MOCK_FROM,
            "EMAIL_PASSWORD": MOCK_PWD,
            "EMAIL_IMAP_SERVER": MOCK_IMAP_SERVER,
            "EMAIL_MARK_AS_SEEN": "False",
        },
    )
    def test_read_emails_mark_as_seen_false(self, mock_imap):
        assert os.getenv("EMAIL_ADDRESS") == MOCK_FROM

        # Create a mock email message
        message = EmailMessage()
        message["From"] = MOCK_FROM
        message["To"] = MOCK_TO
        message["Date"] = MOCK_DATE
        message["Subject"] = MOCK_SUBJECT
        message.set_content(MOCK_CONTENT)

        # Set up mock IMAP server behavior
        mock_imap.return_value.search.return_value = (None, [b"1"])
        mock_imap.return_value.fetch.return_value = (None, [(b"1", message.as_bytes())])

        # Test read_emails function
        result = read_emails("inbox", "UNSEEN")
        expected_result = [
            {
                "From": MOCK_FROM,
                "To": MOCK_TO,
                "Date": MOCK_DATE,
                "CC": "",
                "Subject": MOCK_SUBJECT,
                "Message Body": MOCK_CONTENT,
            }
        ]
        assert result == expected_result

        # Check if the IMAP object was created and used correctly
        mock_imap.return_value.login.assert_called_once_with(MOCK_FROM, MOCK_PWD)
        mock_imap.return_value.select.assert_called_once_with("inbox")
        mock_imap.return_value.search.assert_called_once_with(None, "UNSEEN")
        mock_imap.return_value.fetch.assert_called_once_with(b"1", "(BODY.PEEK[])")

    def side_effect_for_open(original_open, file_path, *args, **kwargs):
        if file_path == MOCK_ATTACHMENT_PATH:
            return mock_open(read_data=b"file_content").return_value
        return original_open(file_path, *args, **kwargs)

    original_open = open
    side_effect_with_original_open = partial(side_effect_for_open, original_open)

    # Test for sending emails with EMAIL_DRAFT_MODE_WITH_FOLDER
    @patch("imaplib.IMAP4_SSL")
    @patch.dict(
        os.environ,
        {
            "EMAIL_ADDRESS": MOCK_FROM,
            "EMAIL_PASSWORD": MOCK_PWD,
            "EMAIL_IMAP_SERVER": MOCK_IMAP_SERVER,
            "EMAIL_DRAFT_MODE_WITH_FOLDER": MOCK_DRAFT_FOLDER,
        },
    )
    @patch(f"{__name__}.imap_open")
    @patch("builtins.open", side_effect=side_effect_with_original_open)
    def test_send_emails_with_draft_mode(self, mock_file, mock_imap_open, mock_imap):
        mock_imap_conn = mock_imap_open.return_value
        mock_imap_conn.select.return_value = ("OK", [b"0"])
        mock_imap_conn.append.return_value = ("OK", [b"1"])

        result = send_email_with_attachment_internal(
            MOCK_TO,
            MOCK_SUBJECT,
            MOCK_CONTENT,
            MOCK_ATTACHMENT_PATH,
            MOCK_ATTACHMENT_NAME,
        )
        assert result == f"Email went to {MOCK_DRAFT_FOLDER}!"
        mock_imap.return_value.login.assert_called_once_with(MOCK_FROM, MOCK_PWD)
        mock_imap.return_value.select.assert_called_once_with(MOCK_DRAFT_FOLDER)

        # Get the actual MIME message appended
        mock_imap.return_value.append.assert_called_once()

        append_args, _ = mock_imap.return_value.append.call_args
        actual_mime_msg = append_args[3].decode("utf-8")

        # Check for the presence of relevant information in the MIME message
        assert MOCK_FROM in actual_mime_msg
        assert MOCK_TO in actual_mime_msg
        assert MOCK_SUBJECT in actual_mime_msg
        assert MOCK_CONTENT in actual_mime_msg
        assert MOCK_ATTACHMENT_NAME in actual_mime_msg


if __name__ == "__main__":
    unittest.main()
