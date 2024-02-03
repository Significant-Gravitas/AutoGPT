import functools
import json
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import BinaryIO, Callable, Literal

import charset_normalizer
import docx
import pypdf
import yaml
from bs4 import BeautifulSoup
from pylatexenc.latex2text import LatexNodes2Text

from AFAAS.interfaces.agent.main import BaseAgent
from AFAAS.lib.sdk.logger import AFAASLogger

LOG = AFAASLogger(name=__name__)

from typing import Callable, ParamSpec, TypeVar

P = ParamSpec("P")
T = TypeVar("T")


class ParserStrategy(ABC):
    @abstractmethod
    def read(self, file: BinaryIO) -> str: ...


# Basic text file reading
class TXTParser(ParserStrategy):
    def read(self, file: BinaryIO) -> str:
        charset_match = charset_normalizer.from_bytes(file.read()).best()
        LOG.debug(
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
        data = yaml.load(file, Loader=yaml.FullLoader)
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
    def __init__(self, parser: ParserStrategy):
        self.parser = parser

    def set_parser(self, parser: ParserStrategy) -> None:
        LOG.trace(f"Setting Context Parser to {parser}")
        self.parser = parser

    def decode_file(self, file: BinaryIO) -> str:
        LOG.debug(f"Reading {getattr(file, 'name', 'file')} with parser {self.parser}")
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


def decode_textual_file(file: BinaryIO, ext: str) -> str:
    if not file.readable():
        raise ValueError(f"{repr(file)} is not readable")

    parser = extension_to_parser.get(ext.lower())
    if not parser:
        if is_file_binary_fn(file):
            raise ValueError(f"Unsupported binary file format: {ext}")
        # fallback to txt file parser (to support script and code files loading)
        parser = TXTParser()
    file_context = FileContext(parser)
    return file_context.decode_file(file)


def sanitize_path_arg(
    arg_name: str, make_relative: bool = False
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Sanitizes the specified path (str | Path) argument, resolving it to a Path"""

    def decorator(func: Callable) -> Callable:
        # Get position of path parameter, in case it is passed as a positional argument
        try:
            arg_index = list(func.__annotations__.keys()).index(arg_name)
        except ValueError:
            raise TypeError(
                f"Sanitized parameter '{arg_name}' absent or not annotated"
                f" on function '{func.__name__}'"
            )

        # Get position of agent parameter, in case it is passed as a positional argument
        try:
            agent_arg_index = list(func.__annotations__.keys()).index("agent")
        except ValueError:
            raise TypeError(
                f"Parameter 'agent' absent or not annotated"
                f" on function '{func.__name__}'"
            )

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            LOG.trace(f"Sanitizing arg '{arg_name}' on function '{func.__name__}'")

            # Get Agent from the called function's arguments
            agent = kwargs.get(
                "agent", len(args) > agent_arg_index and args[agent_arg_index]
            )
            if not isinstance(agent, BaseAgent):
                raise RuntimeError("Could not get Agent from decorated command's args")

            # Sanitize the specified path argument, if one is given
            given_path: str | Path | None = kwargs.get(
                arg_name, len(args) > arg_index and args[arg_index] or None
            )
            if given_path:
                if type(given_path) is str:
                    # Fix workspace path from output in docker environment
                    given_path = re.sub(r"^\/workspace", ".", given_path)

                if given_path in {"", "/", "."}:
                    sanitized_path = agent.workspace.root
                else:
                    sanitized_path = agent.workspace.get_path(given_path)

                # Make path relative if possible
                if make_relative and sanitized_path.is_relative_to(
                    agent.workspace.root
                ):
                    sanitized_path = sanitized_path.relative_to(agent.workspace.root)

                if arg_name in kwargs:
                    kwargs[arg_name] = sanitized_path
                else:
                    # args is an immutable tuple; must be converted to a list to update
                    arg_list = list(args)
                    arg_list[arg_index] = sanitized_path
                    args = tuple(arg_list)

            return func(*args, **kwargs)

        return wrapper

    return decorator
