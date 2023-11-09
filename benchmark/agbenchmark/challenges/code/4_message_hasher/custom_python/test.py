import unittest

from message_hasher import hash_message, retrieve_message


class TestMessageHasher(unittest.TestCase):
    def test_message_retrieval(self):
        # Hash the message to get its hashed form
        hashed_message = hash_message("This is a test message")

        # Retrieve the original message using the hashed message directly
        retrieved_message = retrieve_message(hashed_message)

        self.assertEqual(
            retrieved_message,
            "This is a test message",
            "Retrieved message does not match the original!",
        )


if __name__ == "__main__":
    unittest.main()
