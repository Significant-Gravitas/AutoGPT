import os
import unittest

from autogpt.permanent_memory.sqlite3_store import MemoryDB


class TestMemoryDB(unittest.TestCase):
    def setUp(self):
        self.db_filename = "test_db.sqlite3"
        self.db = MemoryDB(self.db_filename)

    def tearDown(self):
        self.db.quit()
        os.remove(self.db_filename)

    def test_overwrite_and_get_session(self):
        self.db.insert("The quick brown fox jumps over the lazy dog")
        self.db.insert("The five boxing wizards jump quickly")

        # Overwrite the second text
        self.db.overwrite(1, "The slow elephant walks carefully")

        # Get the session and verify the texts
        session = self.db.get_session()
        self.assertEqual(len(session), 2)
        self.assertIn("The quick brown fox jumps over the lazy dog", session)
        self.assertIn("The slow elephant walks carefully", session)

        # Overwrite the first text
        self.db.overwrite(0, "The lazy dog jumps over the quick brown fox")

        # Get the session and verify the texts
        session = self.db.get_session()
        self.assertEqual(len(session), 2)
        self.assertIn("The lazy dog jumps over the quick brown fox", session)
        self.assertIn("The slow elephant walks carefully", session)

    def test_delete_memory(self):
        self.db.insert("The quick brown fox jumps over the lazy dog")
        self.db.insert("The five boxing wizards jump quickly")

        # Delete the first text
        self.db.delete_memory(0)

        # Get the session and verify the remaining text
        session = self.db.get_session()
        self.assertEqual(len(session), 1)
        self.assertIn("The five boxing wizards jump quickly", session)

        # Delete the remaining text
        self.db.delete_memory(1)

        # Get the session and verify that it's empty
        session = self.db.get_session()
        self.assertEqual(len(session), 0)


if __name__ == "__main__":
    unittest.main()
