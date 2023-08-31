from sample_code import multiply_int


def test_multiply_int(num: int, multiplier, expected_result: int) -> None:
    result = multiply_int(num, multiplier)
    print(result)
    assert (
        result == expected_result
    ), f"AssertionError: Expected the output to be {expected_result}"


if __name__ == "__main__":
    # create a trivial test that has 4 as the num, and 2 as the multiplier. Make sure to fill in the expected result
    num =
    multiplier =
    expected_result =
    test_multiply_int()
