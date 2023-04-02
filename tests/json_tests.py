import unittest
import os
import sys
# Probably a better way:
sys.path.append(os.path.abspath('../scripts'))
from json_parser import fix_and_parse_json

class TestParseJson(unittest.TestCase):
    def test_valid_json(self):
        # Test that a valid JSON string is parsed correctly
        json_str = '{"name": "John", "age": 30, "city": "New York"}'
        obj = fix_and_parse_json(json_str)
        self.assertEqual(obj, {"name": "John", "age": 30, "city": "New York"})
    
    def test_invalid_json_minor(self):
        # Test that an invalid JSON string can be fixed with gpt
        json_str = '{"name": "John", "age": 30, "city": "New York",}'
        self.assertEqual(fix_and_parse_json(json_str, try_to_fix_with_gpt=False), {"name": "John", "age": 30, "city": "New York"})
    
    def test_invalid_json_major_with_gpt(self):
        # Test that an invalid JSON string raises an error when try_to_fix_with_gpt is False
        json_str = 'BEGIN: "name": "John" - "age": 30 - "city": "New York" :END'
        self.assertEqual(fix_and_parse_json(json_str, try_to_fix_with_gpt=True), {"name": "John", "age": 30, "city": "New York"})

    def test_invalid_json_major_without_gpt(self):
        # Test that a REALLY invalid JSON string raises an error when try_to_fix_with_gpt is False
        json_str = 'BEGIN: "name": "John" - "age": 30 - "city": "New York" :END'
        # Assert that this raises an exception:
        with self.assertRaises(Exception):
            fix_and_parse_json(json_str, try_to_fix_with_gpt=False)


if __name__ == '__main__':
    unittest.main()