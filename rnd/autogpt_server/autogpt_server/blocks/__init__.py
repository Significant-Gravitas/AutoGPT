from autogpt_server.blocks import sample, reddit
from autogpt_server.data.block import Block

AVAILABLE_BLOCKS = {block.id: block() for block in Block.__subclasses__()}

__all__ = ["sample", "reddit", "AVAILABLE_BLOCKS"]
