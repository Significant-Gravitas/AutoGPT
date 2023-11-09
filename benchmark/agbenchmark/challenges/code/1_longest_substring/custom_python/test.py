# mypy: ignore-errors
from typing import List

from sample_code import lengthOfLongestSubstring


def test_three_sum(string: str, expected_result: int) -> None:
    result = lengthOfLongestSubstring(string)
    print(result)
    assert (
        result == expected_result
    ), f"AssertionError: Expected the output to be {expected_result}"


if __name__ == "__main__":
    # test the trivial case with the first three numbers
    string = "abcabcbb"
    expected_result = 3
    test_three_sum(string, expected_result)
    
    string = "bbbbb"
    expected_result = 1
    test_three_sum(string, expected_result)
    
    string = "pwwkew"
    expected_result = 3
    test_three_sum(string, expected_result)

