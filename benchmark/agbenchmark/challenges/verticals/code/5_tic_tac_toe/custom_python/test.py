import subprocess

import pytest


def run_game_with_inputs(inputs):
    # Start the game process
    process = subprocess.Popen(
        ["python", "tic_tac_toe.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # Send the input moves one by one
    output, errors = process.communicate("\n".join(inputs))

    # Print the inputs and outputs
    print("Inputs:\n", "\n".join(inputs))
    print("Output:\n", output)
    print("Errors:\n", errors)

    return output


@pytest.mark.parametrize(
    "inputs, expected_output",
    [
        (["0,0", "1,0", "0,1", "1,1", "0,2"], "Player 1 won!"),
        (["1,0", "0,0", "1,1", "0,1", "2,0", "0,2"], "Player 2 won!"),
        (["0,0", "0,1", "0,2", "1,1", "1,0", "1,2", "2,1", "2,0", "2,2"], "Draw"),
    ],
)
def test_game(inputs, expected_output):
    output = run_game_with_inputs(inputs)
    assert expected_output in output


if __name__ == "__main__":
    pytest.main([__file__])
