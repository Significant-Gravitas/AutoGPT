"""What a block is called where people read it (Home's Recent work)."""

from backend.blocks.airtable.schema import AirtableCreateTableBlock


def test_an_undeclared_block_reads_the_way_the_builder_titles_it() -> None:
    # beautifyString(name).replace(/ Block$/, "") in NewBlockMenu/Block.tsx.
    assert AirtableCreateTableBlock().display_name == "Airtable Create Table"


def test_a_declared_name_wins() -> None:
    # What ``Block.__init__(display_name=...)`` stores. Set on an instance so
    # the test adds no Block subclass for the registry to pick up.
    block = AirtableCreateTableBlock()
    block._display_name = "Create Airtable table"

    assert block.display_name == "Create Airtable table"
