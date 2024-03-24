from sample_code import multiply_int


def test_multiply_int(num: int, expected_result: int) -> None:
    result = multiply_int(num)
    print(result)
    assert (
        result == expected_result
    ), f"AssertionError: Expected the output to be {expected_result}"


if __name__ == "__main__":
    # test the trivial case
    num = 4
    expected_result = 8
    test_multiply_int(num, expected_result)
