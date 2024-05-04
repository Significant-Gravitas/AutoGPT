from sample_code import multiply_int


def test_multiply_int(num: int, multiplier, expected_result: int) -> None:
    result = multiply_int(num, multiplier)
    print(result)
    assert (
        result == expected_result
    ), f"AssertionError: Expected the output to be {expected_result}"


if __name__ == "__main__":
    # test the trivial case
    num = 4
    multiplier = 2
    expected_result = 8
    test_multiply_int(num, multiplier, expected_result)

    # so it's not hard coded
    num = 7
    multiplier = 7
    expected_result = 49
    test_multiply_int(num, multiplier, expected_result)

    # negative numbers
    num = -6
    multiplier = 2
    expected_result = -12
    test_multiply_int(num, multiplier, expected_result)
