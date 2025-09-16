import pytest

from backend.blocks.table_input import TableInputBlock
from backend.util.test import execute_block_test


@pytest.mark.asyncio
async def test_table_input_block():
    """Test the TableInputBlock with basic input/output."""
    block = TableInputBlock()
    await execute_block_test(block)


@pytest.mark.asyncio
async def test_table_input_with_data():
    """Test TableInputBlock with actual table data."""
    block = TableInputBlock()
    
    input_data = block.Input(
        name="test_table",
        headers=["Name", "Age", "City"],
        value=[
            {"Name": "John", "Age": 30, "City": "New York"},
            {"Name": "Jane", "Age": 25, "City": "London"},
            {"Name": "Bob", "Age": 35, "City": "Paris"}
        ]
    )
    
    output_data = []
    async for output_name, output_value in block.run(input_data):
        output_data.append((output_name, output_value))
    
    assert len(output_data) == 1
    assert output_data[0][0] == "result"
    
    result = output_data[0][1]
    assert len(result) == 3
    assert result[0]["Name"] == "John"
    assert result[1]["Age"] == 25
    assert result[2]["City"] == "Paris"


@pytest.mark.asyncio
async def test_table_input_empty_data():
    """Test TableInputBlock with empty data."""
    block = TableInputBlock()
    
    input_data = block.Input(
        name="empty_table",
        headers=["Col1", "Col2"],
        value=[]
    )
    
    output_data = []
    async for output_name, output_value in block.run(input_data):
        output_data.append((output_name, output_value))
    
    assert len(output_data) == 1
    assert output_data[0][0] == "result"
    assert output_data[0][1] == []


@pytest.mark.asyncio
async def test_table_input_normalization():
    """Test TableInputBlock normalizes missing columns."""
    block = TableInputBlock()
    
    input_data = block.Input(
        name="partial_table",
        headers=["Name", "Age", "City"],
        value=[
            {"Name": "John", "Age": 30},  # Missing City
            {"Name": "Jane", "City": "London"},  # Missing Age
            {"Age": 35, "City": "Paris"}  # Missing Name
        ]
    )
    
    output_data = []
    async for output_name, output_value in block.run(input_data):
        output_data.append((output_name, output_value))
    
    result = output_data[0][1]
    assert len(result) == 3
    
    # Check all headers are present in each row with defaults for missing
    for row in result:
        assert "Name" in row
        assert "Age" in row
        assert "City" in row
    
    assert result[0]["City"] == ""  # Missing City gets empty string
    assert result[1]["Age"] == ""  # Missing Age gets empty string
    assert result[2]["Name"] == ""  # Missing Name gets empty string


@pytest.mark.asyncio
async def test_table_input_empty_headers():
    """Test TableInputBlock with empty headers returns error."""
    block = TableInputBlock()
    
    input_data = block.Input(
        name="no_headers",
        headers=[],
        value=[{"col1": "val1"}]
    )
    
    output_data = []
    async for output_name, output_value in block.run(input_data):
        output_data.append((output_name, output_value))
    
    assert len(output_data) == 1
    assert output_data[0][0] == "error"
    assert "Headers cannot be empty" in output_data[0][1]


@pytest.mark.asyncio
async def test_table_input_duplicate_headers():
    """Test TableInputBlock with duplicate headers returns error."""
    block = TableInputBlock()
    
    input_data = block.Input(
        name="duplicate_headers",
        headers=["Name", "Age", "Name"],  # Duplicate "Name"
        value=[{"Name": "John", "Age": 30}]
    )
    
    output_data = []
    async for output_name, output_value in block.run(input_data):
        output_data.append((output_name, output_value))
    
    assert len(output_data) == 1
    assert output_data[0][0] == "error"
    assert "Duplicate headers are not allowed" in output_data[0][1]