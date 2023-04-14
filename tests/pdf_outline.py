import os
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from io import StringIO

"""
author wangqi
"""
class Outline:
    def __init__(self, title, level, content):
        self.title = title
        self.level = level
        self.content = content

    def __str__(self):
        return f"{'#' * self.level} {self.title}\n\n{self.content}"


def extract_text_from_page(document, page_number):
    rsrcmgr = PDFResourceManager()
    fake_file = StringIO()
    device = TextConverter(rsrcmgr, fake_file, laparams=LAParams())
    interpreter = PDFPageInterpreter(rsrcmgr, device)

    page = list(PDFPage.create_pages(document))[page_number - 1]
    interpreter.process_page(page)
    content = fake_file.getvalue()

    device.close()
    fake_file.close()

    return content.strip()


def extract_outlines(file_path):
    with open(file_path, 'rb') as file:
        parser = PDFParser(file)
        document = PDFDocument(parser)

        if not document.is_extractable:
            raise Exception("The PDF is not extractable.")

        outlines = []

        try:
            raw_outlines = document.get_outlines()
            for level, title, dest, _, _ in raw_outlines:
                if dest is not None and 'PageNumber' in dest:
                    page_number = dest['PageNumber']
                    content = extract_text_from_page(document, page_number)
                else:
                    content = ""

                outline = Outline(title, level, content)
                outlines.append(outline)

        except Exception as e:
            print("No outline found in the PDF.")

        return outlines


if __name__ == "__main__":
    file_path = os.path.join("auto_gpt_workspace", "agent.pdf")
    outlines = extract_outlines(file_path)

    for outline in outlines:
        print(outline)
        print("\n" + "=" * 40 + "\n")
