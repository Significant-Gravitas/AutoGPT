import unittest
import password_generator

class TestPasswordGenerator(unittest.TestCase):
    def test_password_length(self):
        for i in range(10, 21):
            password = password_generator.generate_strong_password(i)
            self.assertEqual(len(password), i, f"Failed for length {i}")

    def test_value_error(self):
        with self.assertRaises(ValueError):
            password_generator.generate_strong_password(9)
        with self.assertRaises(ValueError):
            password_generator.generate_strong_password(21)

    def test_password_content(self):
        for _ in range(100):  # Run the test multiple times to account for randomness
            password = password_generator.generate_strong_password(15)
            self.assertGreaterEqual(sum(c.islower() for c in password), 2)
            self.assertGreaterEqual(sum(c.isupper() for c in password), 2)
            self.assertGreaterEqual(sum(c.isdigit() for c in password), 2)
            self.assertGreaterEqual(sum(c in password_generator.string.punctuation for c in password), 2)

if __name__ == "__main__":
    unittest.main()
