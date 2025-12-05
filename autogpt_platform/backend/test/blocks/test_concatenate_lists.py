"""
Test for ConcatenateListsBlock.

This test verifies the ConcatenateListsBlock functionality.
"""

import pytest

from backend.blocks.data_manipulation import ConcatenateListsBlock


@pytest.mark.asyncio
async def test_concatenate_lists_block_builtin_tests():
    """Test ConcatenateListsBlock using the built-in test_input and test_output."""
    block = ConcatenateListsBlock()
    
    # Verify test data is defined
    assert block.test_input is not None
    assert block.test_output is not None
    
    # Test the first test case from test_input/test_output
    # This matches: {"lists": [[1, 2, 3], [4, 5, 6]]} -> ("concatenated_list", [1, 2, 3, 4, 5, 6])
    input_data = block.Input(**block.test_input[0])
    expected_name, expected_data = block.test_output[0]
    
    result = []
    async for output_name, output_data in block.run(input_data):
        result.append((output_name, output_data))
    
    assert len(result) == 1
    assert result[0][0] == expected_name
    assert result[0][1] == expected_data


@pytest.mark.asyncio
async def test_concatenate_lists_manual():
    """Manual test cases for ConcatenateListsBlock."""
    block = ConcatenateListsBlock()
    
    # Test case 1: Basic concatenation
    input_data = block.Input(lists=[[1, 2, 3], [4, 5, 6]])
    result = []
    async for output_name, output_data in block.run(input_data):
        result.append((output_name, output_data))
    
    assert len(result) == 1
    assert result[0][0] == "concatenated_list"
    assert result[0][1] == [1, 2, 3, 4, 5, 6]
    
    # Test case 2: Empty lists
    input_data = block.Input(lists=[[], [1, 2], []])
    result = []
    async for output_name, output_data in block.run(input_data):
        result.append((output_name, output_data))
    
    assert len(result) == 1
    assert result[0][0] == "concatenated_list"
    assert result[0][1] == [1, 2]
    
    # Test case 3: Single list
    input_data = block.Input(lists=[[1, 2, 3]])
    result = []
    async for output_name, output_data in block.run(input_data):
        result.append((output_name, output_data))
    
    assert len(result) == 1
    assert result[0][0] == "concatenated_list"
    assert result[0][1] == [1, 2, 3]
    
    # Test case 4: Empty input
    input_data = block.Input(lists=[])
    result = []
    async for output_name, output_data in block.run(input_data):
        result.append((output_name, output_data))
    
    assert len(result) == 1
    assert result[0][0] == "concatenated_list"
    assert result[0][1] == []

