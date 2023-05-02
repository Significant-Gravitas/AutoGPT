from autogpt.llm import llm_utils


def test_chunked_tokens():
    text = "Auto-GPT is an experimental open-source application showcasing the capabilities of the GPT-4 language model"
    expected_output = [
        (
            13556,
            12279,
            2898,
            374,
            459,
            22772,
            1825,
            31874,
            3851,
            67908,
            279,
            17357,
            315,
            279,
            480,
            2898,
            12,
            19,
            4221,
            1646,
        )
    ]
    output = list(llm_utils.chunked_tokens(text, "cl100k_base", 8191))
    assert output == expected_output
