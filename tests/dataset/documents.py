import pytest
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

async def load_and_chunk_file(file_path: str):
    if isinstance(file_path, Path):
        file_path = str(file_path)
    loader = TextLoader(file_path)
    text = loader.load()

    text_splitter = RecursiveCharacterTextSplitter()
    chunks = text_splitter.split_documents(text)

    return chunks[:52]


@pytest.fixture(scope="function")
async def documents():
    return await load_and_chunk_file(Path(__file__).parent / "bible.txt")
