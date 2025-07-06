import os
import fitz # PyMuPDF for PDFs
import pdfplumber # Alternative for PDFs, sometimes better with tables
import pytesseract # For OCR (pip install pytesseract, and install Tesseract-OCR engine separately)
from PIL import Image # Pillow, for pytesseract
# from pdf2image import convert_from_path # Uncomment if using pdf2image for OCR (pip install pdf2image, and install poppler on system)
import concurrent.futures
import json
from tqdm import tqdm # For progress bars
import mimetypes # To guess file type
import yaml # To read config.yaml

# Resolve project root (assuming script is in scripts/ directory)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(PROJECT_ROOT, "..")

# --- Configuration Loading ---
def load_config(config_path="config.yaml"):
    """Loads configuration from the project's config.yaml file."""
    full_config_path = os.path.join(PROJECT_ROOT, config_path)
    with open(full_config_path, 'r') as f:
        return yaml.safe_load(f)

CONFIG = load_config()

# Define paths from config
RAW_TEXT_DIR = os.path.join(PROJECT_ROOT, CONFIG['data_settings']['raw_text_dir'])
RAW_PDF_DIR = os.path.join(PROJECT_ROOT, CONFIG['data_settings']['raw_pdf_dir'])
RAW_CODE_DIR = os.path.join(PROJECT_ROOT, CONFIG['data_settings']['raw_code_dir'])
PROCESSED_DOCS_PATH = os.path.join(PROJECT_ROOT, CONFIG['data_settings']['processed_extracted_raw_documents'])

# --- Text Extraction Functions ---
def extract_text_from_generic_text_file(filepath: str) -> str | None:
    """Extracts text content from a generic text-based file (e.g., .txt, .md, code files)."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading text file {filepath}: {e}")
        return None

def extract_text_from_pdf(filepath: str, use_ocr: bool = False) -> str | None:
    """
    Extracts text content from a PDF file.
    Attempts direct text extraction first, falls back to pdfplumber, and optionally to OCR.
    """
    text_content = ""
    try:
        # Attempt direct text extraction with PyMuPDF first (best for selectable text PDFs)
        doc = fitz.open(filepath)
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text_content += page.get_text() + "\n"
        doc.close()
        if text_content.strip(): # Check if any meaningful text was extracted
            return text_content
    except Exception as e:
        print(f"PyMuPDF failed for {filepath}: {e}. Trying pdfplumber...")
        try: # Fallback to pdfplumber
            with pdfplumber.open(filepath) as pdf:
                for page in pdf.pages:
                    text_content += page.extract_text() or "" # extract_text can return None
            if text_content.strip():
                return text_content
        except Exception as e_plumber:
            print(f"pdfplumber also failed for {filepath}: {e_plumber}.")

    # If direct extraction failed and OCR is enabled
    if use_ocr:
        print(f"Attempting OCR for {filepath} as direct extraction failed...")
        try:
            # Requires `pdf2image` and `poppler` on system to convert PDF to image for robust OCR
            # from pdf2image import convert_from_path
            # images = convert_from_path(filepath) # This line requires poppler to be installed
            # ocr_text = ""
            # for img in images:
            #     ocr_text += pytesseract.image_to_string(img) + "\n"
            # return ocr_text

            # Simpler OCR using PyMuPDF's pixmap directly with PIL and pytesseract
            doc = fitz.open(filepath)
            ocr_text = ""
            for page_num in range(doc.page_count):
                page = doc.load_page(page_num)
                pix = page.get_pixmap()
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                ocr_text += pytesseract.image_to_string(img) + "\n"
            doc.close()
            if ocr_text.strip():
                return ocr_text
            else:
                print(f"OCR extracted no meaningful text from {filepath}.")
                return None
        except Exception as e_ocr:
            print(f"OCR failed for {filepath}: {e_ocr}. Ensure Tesseract-OCR is installed and configured.")
            return None
    
    print(f"Text extraction failed for {filepath} (no OCR or OCR failed).")
    return None # Failed to extract any text

# --- File Discovery ---
def get_files_in_folder(folder_path: str, allowed_extensions: list[str]) -> list[str]:
    """Recursively gets all files in a folder matching allowed extensions."""
    files = []
    for root, _, filenames in os.walk(folder_path):
        for filename in filenames:
            _, ext = os.path.splitext(filename)
            if ext.lower() in allowed_extensions:
                files.append(os.path.join(root, filename))
    return files

# --- Main Ingestion Logic ---
def ingest_raw_data_pipeline(use_ocr_for_pdfs: bool = False, max_workers: int = os.cpu_count()) -> list[dict]:
    """
    Orchestrates the raw data ingestion process: discovers files, extracts text, and saves metadata.
    Args:
        use_ocr_for_pdfs (bool): Whether to attempt OCR for PDF files if direct text extraction fails.
        max_workers (int): Number of threads for parallel file processing.
    Returns:
        list[dict]: A list of dictionaries, each representing an extracted document.
    """
    print("Starting raw data ingestion pipeline...")

    # Define allowed extensions for each raw data directory
    # Ensure these lists are comprehensive for your data
    text_extensions = [".txt", ".md"]
    code_extensions = [".py", ".js", ".java", ".cpp", ".c", ".h", ".cs", ".go", ".rs", ".ts", ".php", ".html", ".css", ".xml", ".json", ".yaml", ".yml", ".sql", ".sh"]
    pdf_extensions = [".pdf"]

    all_raw_files = []
    all_raw_files.extend(get_files_in_folder(RAW_TEXT_DIR, text_extensions))
    all_raw_files.extend(get_files_in_folder(RAW_CODE_DIR, code_extensions))
    all_raw_files.extend(get_files_in_folder(RAW_PDF_DIR, pdf_extensions))

    extracted_documents = []
    skipped_files = []

    def process_single_file_wrapper(filepath: str) -> dict | None:
        """Wrapper to process a single file and return its extracted data and metadata."""
        content = None
        file_type = "unknown"
        
        _, ext = os.path.splitext(filepath)
        ext = ext.lower()

        if ext in pdf_extensions:
            file_type = "pdf"
            content = extract_text_from_pdf(filepath, use_ocr=use_ocr_for_pdfs)
        elif ext in text_extensions:
            file_type = "text"
            content = extract_text_from_generic_text_file(filepath)
        elif ext in code_extensions:
            file_type = "code"
            content = extract_text_from_generic_text_file(filepath)
        else:
            print(f"Skipping unsupported file type: {filepath}")
            skipped_files.append(filepath)
            return None

        if content is not None: # Check for None explicitly, as empty string is valid content
            return {
                "content": content,
                "source_filename": os.path.basename(filepath),
                "source_filepath": os.path.relpath(filepath, PROJECT_ROOT), # Path relative to project root
                "file_type": file_type, # e.g., 'text', 'code', 'pdf'
                "last_modified": os.path.getmtime(filepath) # Unix timestamp
            }
        else:
            skipped_files.append(filepath)
            return None

    # Use ThreadPoolExecutor for I/O bound tasks like file reading
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(process_single_file_wrapper, all_raw_files), total=len(all_raw_files), desc="Extracting text from raw files"))

    for res in results:
        if res:
            extracted_documents.append(res)
        # Skipped files are handled inside process_single_file_wrapper

    os.makedirs(os.path.dirname(PROCESSED_DOCS_PATH), exist_ok=True)
    with open(PROCESSED_DOCS_PATH, 'w', encoding='utf-8') as f:
        for doc in extracted_documents:
            f.write(json.dumps(doc) + "\n")
    print(f"Extracted content from {len(extracted_documents)} files and saved to {PROCESSED_DOCS_PATH}")
    if skipped_files:
        print(f"Skipped {len(skipped_files)} files due to errors or unsupported types.")
    return extracted_documents

if __name__ == "__main__":
    # Example usage:
    # To run with basic OCR for PDFs (requires Tesseract-OCR installed and configured)
    # Be aware that full OCR setup (pdf2image + poppler) is more involved.
    ingest_raw_data_pipeline(use_ocr_for_pdfs=False)
    # You can change use_ocr_for_pdfs to True if you have Tesseract-OCR installed.
