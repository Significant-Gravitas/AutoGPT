import pytest

from backend.blocks.io import AgentTableInputBlock
from backend.util.test import execute_block_test


@pytest.mark.asyncio
async def test_table_input_block():
    """Test the AgentTableInputBlock with basic input/output."""
    block = AgentTableInputBlock()
    await execute_block_test(block)


@pytest.mark.asyncio
async def test_table_input_with_data():
    """Test AgentTableInputBlock with actual table data."""
    block = AgentTableInputBlock()

    input_data = block.Input(
        name="test_table",
        column_headers=["Name", "Age", "City"],
        value=[
            {"Name": "John", "Age": "30", "City": "New York"},
            {"Name": "Jane", "Age": "25", "City": "London"},
            {"Name": "Bob", "Age": "35", "City": "Paris"},
        ],
    )

    output_data = []
    async for output_name, output_value in block.run(input_data):
        output_data.append((output_name, output_value))

    assert len(output_data) == 1
    assert output_data[0][0] == "result"

    result = output_data[0][1]
    assert len(result) == 3
    assert result[0]["Name"] == "John"
    assert result[1]["Age"] == "25"
    assert result[2]["City"] == "Paris"


@pytest.mark.asyncio
async def test_table_input_empty_data():
    """Test AgentTableInputBlock with empty data."""
    block = AgentTableInputBlock()

    input_data = block.Input(
        name="empty_table", column_headers=["Col1", "Col2"], value=[]
    )

    output_data = []
    async for output_name, output_value in block.run(input_data):
        output_data.append((output_name, output_value))

    assert len(output_data) == 1
    assert output_data[0][0] == "result"
    assert output_data[0][1] == []


@pytest.mark.asyncio
async def test_table_input_with_missing_columns():
    """Test AgentTableInputBlock passes through data with missing columns as-is."""
    block = AgentTableInputBlock()

    input_data = block.Input(
        name="partial_table",
        column_headers=["Name", "Age", "City"],
        value=[
            {"Name": "John", "Age": "30"},  # Missing City
            {"Name": "Jane", "City": "London"},  # Missing Age
            {"Age": "35", "City": "Paris"},  # Missing Name
        ],
    )

    output_data = []
    async for output_name, output_value in block.run(input_data):
        output_data.append((output_name, output_value))

    result = output_data[0][1]
    assert len(result) == 3

    # Check data is passed through as-is
    assert result[0] == {"Name": "John", "Age": "30"}
    assert result[1] == {"Name": "Jane", "City": "London"}
    assert result[2] == {"Age": "35", "City": "Paris"}


@pytest.mark.asyncio
async def test_table_input_none_value():
    """Test AgentTableInputBlock with None value returns empty list."""
    block = AgentTableInputBlock()

    input_data = block.Input(
        name="none_table", column_headers=["Name", "Age"], value=None
    )

    output_data = []
    async for output_name, output_value in block.run(input_data):
        output_data.append((output_name, output_value))

    assert len(output_data) == 1
    assert output_data[0][0] == "result"
    assert output_data[0][1] == []


@pytest.mark.asyncio
async def test_table_input_with_default_headers():
    """Test AgentTableInputBlock with default column headers."""
    block = AgentTableInputBlock()

    # Don't specify column_headers, should use defaults
    input_data = block.Input(
        name="default_headers_table",
        value=[
            {"Column 1": "A", "Column 2": "B", "Column 3": "C"},
            {"Column 1": "D", "Column 2": "E", "Column 3": "F"},
        ],
    )

    output_data = []
    async for output_name, output_value in block.run(input_data):
        output_data.append((output_name, output_value))

    assert len(output_data) == 1
    assert output_data[0][0] == "result"

    result = output_data[0][1]
    assert len(result) == 2
    assert result[0]["Column 1"] == "A"
    assert result[1]["Column 3"] == "F"
