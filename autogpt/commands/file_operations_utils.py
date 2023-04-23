import os
import PyPDF2
import docx
import json
import yaml
from bs4 import BeautifulSoup
import markdown
from pylatexenc.latex2text import LatexNodes2Text


class ParserStrategy:
    def read(self, file_path):
        raise NotImplementedError


# Basic text file reading
class TXTParser(ParserStrategy):
    def read(self, file_path):
        with open(file_path, "r") as f:
            text = f.read()
        return text


# Reading text from binary file using pdf parser
class PDFParser(ParserStrategy):
    def read(self, file_path):
        parser = PyPDF2.PdfReader(file_path)
        text = ""
        for page_idx in range(len(parser.pages)):
            text += parser.pages[page_idx].extract_text()
        return text


# Reading text from binary file using docs parser
class DOCXParser(ParserStrategy):
    def read(self, file_path):
        doc_file = docx.Document(file_path)
        text = ""
        for para in doc_file.paragraphs:
            text += para.text
        return text


# Reading as dictionary and returning string format
class JSONParser(ParserStrategy):
    def read(self, file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
            text = str(data)
        return text


class XMLParser(ParserStrategy):
    def read(self, file_path):
        with open(file_path, "r") as f:
            soup = BeautifulSoup(f, "xml")
            text = soup.get_text()
        return text


# Reading as dictionary and returning string format
class YAMLParser(ParserStrategy):
    def read(self, file_path):
        with open(file_path, "r") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
            text = str(data)
        return text


class HTMLParser(ParserStrategy):
    def read(self, file_path):
        with open(file_path, "r") as f:
            soup = BeautifulSoup(f, "html.parser")
            text = soup.get_text()
        return text


class MarkdownParser(ParserStrategy):
    def read(self, file_path):
        with open(file_path, "r") as f:
            html = markdown.markdown(f.read())
            text = "".join(BeautifulSoup(html, "html.parser").findAll(string=True))
        return text


class LaTeXParser(ParserStrategy):
    def read(self, file_path):
        with open(file_path, "r") as f:
            latex = f.read()
        text = LatexNodes2Text().latex_to_text(latex)
        return text


class FileContext:
    def __init__(self, parser):
        self.parser = parser

    def set_parser(self, parser):
        self.parser = parser

    def read_file(self, file_path):
        return self.parser.read(file_path)


extension_to_parser = {
    ".txt": TXTParser(),
    ".csv": TXTParser(),
    ".pdf": PDFParser(),
    ".doc": DOCXParser(),
    ".docx": DOCXParser(),
    ".json": JSONParser(),
    ".xml": XMLParser(),
    ".yaml": YAMLParser(),
    ".html": HTMLParser(),
    ".md": MarkdownParser(),
    ".tex": LaTeXParser(),
}


def read_textual_file(file_path):
    file_extension = os.path.splitext(file_path)[1].lower()
    parser = extension_to_parser.get(file_extension)
    if not parser:
        # either fallback to txt file parser
        # parser = TXTParser()
        # or raise exception
        raise ValueError("Unsupported file format: {}".format(file_extension))
    file_context = FileContext(parser)
    return file_context.read_file(file_path)
