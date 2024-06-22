import json
import logging
import os.path
import tempfile
from pathlib import Path
from xml.etree import ElementTree

import docx
import pytest
import yaml
from bs4 import BeautifulSoup

from .file_operations import decode_textual_file, is_file_binary_fn

logger = logging.getLogger(__name__)

plain_text_str = "Hello, world!"


def mock_text_file():
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        f.write(plain_text_str)
    return f.name


def mock_csv_file():
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
        f.write(plain_text_str)
    return f.name


def mock_pdf_file():
    with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".pdf") as f:
        # Create a new PDF and add a page with the text plain_text_str
        # Write the PDF header
        f.write(b"%PDF-1.7\n")
        # Write the document catalog
        f.write(b"1 0 obj\n")
        f.write(b"<< /Type /Catalog /Pages 2 0 R >>\n")
        f.write(b"endobj\n")
        # Write the page object
        f.write(b"2 0 obj\n")
        f.write(
            b"<< /Type /Page /Parent 1 0 R /Resources << /Font << /F1 3 0 R >> >> "
            b"/MediaBox [0 0 612 792] /Contents 4 0 R >>\n"
        )
        f.write(b"endobj\n")
        # Write the font object
        f.write(b"3 0 obj\n")
        f.write(
            b"<< /Type /Font /Subtype /Type1 /Name /F1 /BaseFont /Helvetica-Bold >>\n"
        )
        f.write(b"endobj\n")
        # Write the page contents object
        f.write(b"4 0 obj\n")
        f.write(b"<< /Length 25 >>\n")
        f.write(b"stream\n")
        f.write(b"BT\n/F1 12 Tf\n72 720 Td\n(Hello, world!) Tj\nET\n")
        f.write(b"endstream\n")
        f.write(b"endobj\n")
        # Write the cross-reference table
        f.write(b"xref\n")
        f.write(b"0 5\n")
        f.write(b"0000000000 65535 f \n")
        f.write(b"0000000017 00000 n \n")
        f.write(b"0000000073 00000 n \n")
        f.write(b"0000000123 00000 n \n")
        f.write(b"0000000271 00000 n \n")
        f.write(b"trailer\n")
        f.write(b"<< /Size 5 /Root 1 0 R >>\n")
        f.write(b"startxref\n")
        f.write(b"380\n")
        f.write(b"%%EOF\n")
        f.write(b"\x00")
    return f.name


def mock_docx_file():
    with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".docx") as f:
        document = docx.Document()
        document.add_paragraph(plain_text_str)
        document.save(f.name)
    return f.name


def mock_json_file():
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".json") as f:
        json.dump({"text": plain_text_str}, f)
    return f.name


def mock_xml_file():
    root = ElementTree.Element("text")
    root.text = plain_text_str
    tree = ElementTree.ElementTree(root)
    with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".xml") as f:
        tree.write(f)
    return f.name


def mock_yaml_file():
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".yaml") as f:
        yaml.dump({"text": plain_text_str}, f)
    return f.name


def mock_html_file():
    html = BeautifulSoup(
        "<html>"
        "<head><title>This is a test</title></head>"
        f"<body><p>{plain_text_str}</p></body>"
        "</html>",
        "html.parser",
    )
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".html") as f:
        f.write(str(html))
    return f.name


def mock_md_file():
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".md") as f:
        f.write(f"# {plain_text_str}!\n")
    return f.name


def mock_latex_file():
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".tex") as f:
        latex_str = (
            r"\documentclass{article}"
            r"\begin{document}"
            f"{plain_text_str}"
            r"\end{document}"
        )
        f.write(latex_str)
    return f.name


respective_file_creation_functions = {
    ".txt": mock_text_file,
    ".csv": mock_csv_file,
    ".pdf": mock_pdf_file,
    ".docx": mock_docx_file,
    ".json": mock_json_file,
    ".xml": mock_xml_file,
    ".yaml": mock_yaml_file,
    ".html": mock_html_file,
    ".md": mock_md_file,
    ".tex": mock_latex_file,
}
binary_files_extensions = [".pdf", ".docx"]


@pytest.mark.parametrize(
    "file_extension, c_file_creator",
    respective_file_creation_functions.items(),
)
def test_parsers(file_extension, c_file_creator):
    created_file_path = Path(c_file_creator())
    with open(created_file_path, "rb") as file:
        loaded_text = decode_textual_file(file, os.path.splitext(file.name)[1], logger)

        assert plain_text_str in loaded_text

        should_be_binary = file_extension in binary_files_extensions
        assert should_be_binary == is_file_binary_fn(file)

    created_file_path.unlink()  # cleanup
