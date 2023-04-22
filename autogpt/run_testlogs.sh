#!/bin/bash

# Run the testlogs.py script
python testlogs.py


# Compile the .typ file to a PDF using typst
typst compile ../logs/log.typ

# Define the PDF file name (assuming it has the same name as the .typ file with a .pdf extension)
pdf_file="../logs/log.pdf"

# Open the PDF file with the default viewer
case "$(uname -s)" in
    Linux*)     xdg-open "$pdf_file" ;;
    Darwin*)    open "$pdf_file" ;;
    CYGWIN*|MINGW32*|MSYS*|MINGW*) start "$pdf_file" ;;
    *) echo "Unsupported OS" ;;
esac