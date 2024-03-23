import json
import logging
from abc import ABC, abstractmethod
from typing import BinaryIO

import charset_normalizer
import docx
import pypdf
import yaml
from bs4 import BeautifulSoup
from pylatexenc.latex2text import LatexNodes2Text

logger = logging.getLogger(__name__)


class ParserStrategy(ABC):
    @abstractmethod
    def read(self, file: BinaryIO) -> str:
        ...


# Basic text file reading
class TXTParser(ParserStrategy):
    def read(self, file: BinaryIO) -> str:
        charset_match = charset_normalizer.from_bytes(file.read()).best()
        logger.debug(
            f"Reading {getattr(file, 'name', 'file')} "
            f"with encoding '{charset_match.encoding}'"
        )
        return str(charset_match)


# Reading text from binary file using pdf parser
class PDFParser(ParserStrategy):
    def read(self, file: BinaryIO) -> str:
        parser = pypdf.PdfReader(file)
        text = ""
        for page_idx in range(len(parser.pages)):
            text += parser.pages[page_idx].extract_text()
        return text


# Reading text from binary file using docs parser
class DOCXParser(ParserStrategy):
    def read(self, file: BinaryIO) -> str:
        doc_file = docx.Document(file)
        text = ""
        for para in doc_file.paragraphs:
            text += para.text
        return text


# Reading as dictionary and returning string format
class JSONParser(ParserStrategy):
    def read(self, file: BinaryIO) -> str:
        data = json.load(file)
        text = str(data)
        return text


class XMLParser(ParserStrategy):
    def read(self, file: BinaryIO) -> str:
        soup = BeautifulSoup(file, "xml")
        text = soup.get_text()
        return text


# Reading as dictionary and returning string format
class YAMLParser(ParserStrategy):
    def read(self, file: BinaryIO) -> str:
        data = yaml.load(file, Loader=yaml.SafeLoader)
        text = str(data)
        return text


class HTMLParser(ParserStrategy):
    def read(self, file: BinaryIO) -> str:
        soup = BeautifulSoup(file, "html.parser")
        text = soup.get_text()
        return text


class LaTeXParser(ParserStrategy):
    def read(self, file: BinaryIO) -> str:
        latex = file.read().decode()
        text = LatexNodes2Text().latex_to_text(latex)
        return text


class FileContext:
    def __init__(self, parser: ParserStrategy, logger: logging.Logger):
        self.parser = parser
        self.logger = logger

    def set_parser(self, parser: ParserStrategy) -> None:
        self.logger.debug(f"Setting Context Parser to {parser}")
        self.parser = parser

    def decode_file(self, file: BinaryIO) -> str:
        self.logger.debug(
            f"Reading {getattr(file, 'name', 'file')} with parser {self.parser}"
        )
        return self.parser.read(file)


extension_to_parser = {
    ".txt": TXTParser(),
    ".md": TXTParser(),
    ".markdown": TXTParser(),
    ".csv": TXTParser(),
    ".pdf": PDFParser(),
    ".docx": DOCXParser(),
    ".json": JSONParser(),
    ".xml": XMLParser(),
    ".yaml": YAMLParser(),
    ".yml": YAMLParser(),
    ".html": HTMLParser(),
    ".htm": HTMLParser(),
    ".xhtml": HTMLParser(),
    ".tex": LaTeXParser(),
}


def is_file_binary_fn(file: BinaryIO):
    """Given a file path load all its content and checks if the null bytes is present

    Args:
        file (_type_): _description_

    Returns:
        bool: is_binary
    """
    file_data = file.read()
    file.seek(0)
    if b"\x00" in file_data:
        return True
    return False


def decode_textual_file(file: BinaryIO, ext: str, logger: logging.Logger) -> str:
    if not file.readable():
        raise ValueError(f"{repr(file)} is not readable")

    parser = extension_to_parser.get(ext.lower())
    if not parser:
        if is_file_binary_fn(file):
            raise ValueError(f"Unsupported binary file format: {ext}")
        # fallback to txt file parser (to support script and code files loading)
        parser = TXTParser()
    file_context = FileContext(parser, logger)
    return file_context.decode_file(file)
