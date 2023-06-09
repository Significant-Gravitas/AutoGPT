import json
import os

import charset_normalizer
import docx
import markdown
import PyPDF2
import yaml
from bs4 import BeautifulSoup
from pylatexenc.latex2text import LatexNodes2Text

from autogpt import logs
from autogpt.logs import logger


class ParserStrategy:
    def read(self, file_path: str) -> str:
        raise NotImplementedError


# Basic text file reading
class TXTParser(ParserStrategy):
    def read(self, file_path: str) -> str:
        charset_match = charset_normalizer.from_path(file_path).best()
        logger.debug(f"Reading '{file_path}' with encoding '{charset_match.encoding}'")
        return str(charset_match)


# Reading text from binary file using pdf parser
class PDFParser(ParserStrategy):
    def read(self, file_path: str) -> str:
        parser = PyPDF2.PdfReader(file_path)
        text = ""
        for page_idx in range(len(parser.pages)):
            text += parser.pages[page_idx].extract_text()
        return text


# Reading text from binary file using docs parser
class DOCXParser(ParserStrategy):
    def read(self, file_path: str) -> str:
        doc_file = docx.Document(file_path)
        text = ""
        for para in doc_file.paragraphs:
            text += para.text
        return text


# Reading as dictionary and returning string format
class JSONParser(ParserStrategy):
    def read(self, file_path: str) -> str:
        with open(file_path, "r") as f:
            data = json.load(f)
            text = str(data)
        return text


class XMLParser(ParserStrategy):
    def read(self, file_path: str) -> str:
        with open(file_path, "r") as f:
            soup = BeautifulSoup(f, "xml")
            text = soup.get_text()
        return text


# Reading as dictionary and returning string format
class YAMLParser(ParserStrategy):
    def read(self, file_path: str) -> str:
        with open(file_path, "r") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
            text = str(data)
        return text


class HTMLParser(ParserStrategy):
    def read(self, file_path: str) -> str:
        with open(file_path, "r") as f:
            soup = BeautifulSoup(f, "html.parser")
            text = soup.get_text()
        return text


class MarkdownParser(ParserStrategy):
    def read(self, file_path: str) -> str:
        with open(file_path, "r") as f:
            html = markdown.markdown(f.read())
            text = "".join(BeautifulSoup(html, "html.parser").findAll(string=True))
        return text


class LaTeXParser(ParserStrategy):
    def read(self, file_path: str) -> str:
        with open(file_path, "r") as f:
            latex = f.read()
        text = LatexNodes2Text().latex_to_text(latex)
        return text


class FileContext:
    def __init__(self, parser: ParserStrategy, logger: logs.Logger):
        self.parser = parser
        self.logger = logger

    def set_parser(self, parser: ParserStrategy) -> None:
        self.logger.debug(f"Setting Context Parser to {parser}")
        self.parser = parser

    def read_file(self, file_path) -> str:
        self.logger.debug(f"Reading file {file_path} with parser {self.parser}")
        return self.parser.read(file_path)


extension_to_parser = {
    ".txt": TXTParser(),
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
    ".md": MarkdownParser(),
    ".markdown": MarkdownParser(),
    ".tex": LaTeXParser(),
}


def is_file_binary_fn(file_path: str):
    """Given a file path load all its content and checks if the null bytes is present

    Args:
        file_path (_type_): _description_

    Returns:
        bool: is_binary
    """
    with open(file_path, "rb") as f:
        file_data = f.read()
    if b"\x00" in file_data:
        return True
    return False


def read_textual_file(file_path: str, logger: logs.Logger) -> str:
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"{file_path} not found!")
    is_binary = is_file_binary_fn(file_path)
    file_extension = os.path.splitext(file_path)[1].lower()
    parser = extension_to_parser.get(file_extension)
    if not parser:
        if is_binary:
            raise ValueError(f"Unsupported binary file format: {file_extension}")
        # fallback to txt file parser (to support script and code files loading)
        parser = TXTParser()
    file_context = FileContext(parser, logger)
    return file_context.read_file(file_path)
