from helper import parse_pdf_to_text, parse_pdf_with_llamaparse
from openai import OpenAI
import shutil
from pathlib import Path
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Process PDFs in current directory
# for file in os.listdir("."):
#     if file.endswith(".pdf"):
#         parse_pdf_to_text(file)

# Process PDFs in nonParsedFiles directory with LlamaParse
non_parsed_dir = Path("nonParsedFiles")
if non_parsed_dir.exists():
    for file in os.listdir(non_parsed_dir):
        if file.endswith(".pdf"):
            pdf_path = non_parsed_dir / file
            try:
                parse_pdf_with_llamaparse(str(pdf_path))
            except Exception as e:
                print(f"Error processing {file} with LlamaParse: {e}")