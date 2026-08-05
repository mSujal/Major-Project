"""
    Extract the text from the PDF file.

    Two ways the text is extracted from PDF:
        1. Plain text extracted usinf PyMuPDF4LLM in text form.
        2. OCR using pytesseract if text is not extractable.
"""

import pymupdf4llm 
import pytesseract 
from pdf2image import convert_from_path

class Extraction:
    def __init__(self, pdf_path):
        """
        Args:
            pdf_path: pdf from which the text is to be extracted
        """
        self.pdf_path = pdf_path 

    def _ocr_page(self, page_num):
        """
        Fallback OCR for single page that is not extractable

        Args:
            page_num: Single page in which OCR is to be applied
        """
        images = convert_from_path(self.pdf_path, first_page=page_num, last_page=page_num, dpi=300)
        if not images:
            return "" # Nothing can be done for the page
        return pytesseract.image_to_string(images[0]).strip()   # <-- added parentheses

    def extract_text(self):
        """
        Return list of (page_num, clean_text) tuples, one per page chunk
        """
        page_chunks = pymupdf4llm.to_text(self.pdf_path, page_chunks=True, show_progress=True)

        results = []
        for chunk in page_chunks:
            page_num = chunk["metadata"]["page_number"]
            text = chunk["text"].strip()

            if len(text) < 50: # Fallback to OCR
                text = self._ocr_page(page_num)

            if text: # Text is extracted or OCR returned the text 
                results.append((page_num, text))
        
        return results