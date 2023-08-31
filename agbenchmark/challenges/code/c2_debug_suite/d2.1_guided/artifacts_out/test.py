from typing import List

from sample_code import two_sum


def test_two_sum(nums: List, target: int, expected_result: List[int]) -> None:
    result = two_sum(nums, target)
    print(result)
    assert (
        result == expected_result
    ), f"AssertionError: Expected the output to be {expected_result}"


if __name__ == "__main__":
    # test the trivial case with the first two numbers
    nums = [2, 7, 11, 15]
    target = 9
    expected_result = [0, 1]
    test_two_sum(nums, target, expected_result)

    # test for ability to use zero and the same number twice
    nums = [2, 7, 0, 15, 12, 0]
    target = 0
    expected_result = [2, 5]
    test_two_sum(nums, target, expected_result)

    # test for first and last index usage and negative numbers
    nums = [-6, 7, 11, 4]
    target = -2
    expected_result = [0, 3]
    test_two_sum(nums, target, expected_result)
