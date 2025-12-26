import os
from pathlib import Path


def parse_pdf_to_text(pdf_path, output_path=None):
    """
    Parse a PDF file and extract text to a text file.
    
    Args:
        pdf_path (str): Path to the input PDF file
        output_path (str, optional): Path to the output text file. 
                                     If None, creates a file with the same name as PDF but with .txt extension
                                     in the same directory or in parsedFiles directory if it exists.
    
    Returns:
        str: Path to the created text file
    
    Raises:
        FileNotFoundError: If the PDF file doesn't exist
        ImportError: If required library is not installed
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        try:
            from PyPDF2 import PdfReader
        except ImportError:
            raise ImportError(
                "pypdf or PyPDF2 library is required. Install it using: pip install pypdf"
            )
    
    # Check if PDF file exists
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    # Determine output path
    if output_path is None:
        # Check if parsedFiles directory exists
        parsed_files_dir = Path(pdf_path.parent) / "parsedFiles"
        if parsed_files_dir.exists():
            output_path = parsed_files_dir / f"{pdf_path.stem}.txt"
        else:
            output_path = pdf_path.parent / f"{pdf_path.stem}.txt"
    else:
        output_path = Path(output_path)
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Read PDF and extract text
    try:
        reader = PdfReader(str(pdf_path))
        text_content = []
        
        # Extract text from each page
        for page_num, page in enumerate(reader.pages, start=1):
            try:
                text = page.extract_text()
                if text.strip():  # Only add non-empty pages
                    text_content.append(f"--- Page {page_num} ---\n")
                    text_content.append(text)
                    text_content.append("\n\n")
            except Exception as e:
                print(f"Warning: Could not extract text from page {page_num}: {e}")
                continue
        
        # Write to output file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(''.join(text_content))
        
        print(f"Successfully extracted text from PDF to: {output_path}")
        return str(output_path)
    
    except Exception as e:
        raise Exception(f"Error parsing PDF: {e}")


def parse_pdf_with_llamaparse(pdf_path, output_path=None, api_key=None):
    """
    Parse a PDF file using LlamaParse and extract text to a text file.
    
    Args:
        pdf_path (str): Path to the input PDF file
        output_path (str, optional): Path to the output text file. 
                                     If None, creates a file with the same name as PDF but with .txt extension
                                     in parsedFilesLlama directory.
        api_key (str, optional): LlamaParse API key. If None, will try to get from LLAMA_CLOUD_API_KEY environment variable.
    
    Returns:
        str: Path to the created text file
    
    Raises:
        FileNotFoundError: If the PDF file doesn't exist
        ImportError: If required library is not installed
        ValueError: If API key is not provided
    """
    try:
        from llama_parse import LlamaParse
    except ImportError:
        raise ImportError(
            "llama-parse library is required. Install it using: pip install llama-parse"
        )
    
    import os
    
    # Get API key
    if api_key is None:
        api_key = os.getenv('LLAMA_CLOUD_API_KEY')
        if not api_key:
            raise ValueError(
                "LlamaParse API key is required. "
                "Set LLAMA_CLOUD_API_KEY environment variable or pass api_key parameter."
            )
    
    # Check if PDF file exists
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    # Determine output path
    if output_path is None:
        # Use parsedFilesLlama directory
        base_dir = pdf_path.parent.parent if pdf_path.parent.name in ['nonParsedFiles', 'parsedFiles'] else pdf_path.parent
        parsed_files_dir = base_dir / "parsedFilesLlama"
        parsed_files_dir.mkdir(parents=True, exist_ok=True)
        output_path = parsed_files_dir / f"{pdf_path.stem}.txt"
    else:
        output_path = Path(output_path)
    
    # Create output directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Parse PDF using LlamaParse
    try:
        parser = LlamaParse(
            api_key=api_key,
            result_type="text",  # Get plain text output
            verbose=True
        )
        
        print(f"Parsing PDF with LlamaParse: {pdf_path.name}...")
        documents = parser.load_data(str(pdf_path))
        
        # Extract text from documents
        text_content = []
        for doc in documents:
            if hasattr(doc, 'text'):
                text_content.append(doc.text)
            elif isinstance(doc, str):
                text_content.append(doc)
        
        # Write to output file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(text_content))
        
        print(f"Successfully extracted text from PDF using LlamaParse to: {output_path}")
        return str(output_path)
    
    except Exception as e:
        raise Exception(f"Error parsing PDF with LlamaParse: {e}")


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        pdf_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else None
        parse_pdf_to_text(pdf_file, output_file)
    else:
        print("Usage: python main.py <pdf_path> [output_path]")
        print("Example: python main.py document.pdf")
        print("Example: python main.py document.pdf output.txt")

