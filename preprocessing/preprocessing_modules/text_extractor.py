"""
Text Extractor Module

Handles extracting text content from PDF files.
"""

import pdfplumber


class TextExtractor:
    """Handles text extraction from PDF files."""
    
    def __init__(self):
        """Initialize the text extractor."""
        pass
    
    async def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            str: Extracted text content
            
        Raises:
            Exception: If text extraction fails
        """
        print(f"📖 Extracting text from PDF...")
        
        full_text = ""
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text:
                        full_text += f"\n--- Page {page_num + 1} ---\n"
                        full_text += text
            
            print(f"✅ Extracted {len(full_text)} characters from PDF")
            return full_text
            
        except Exception as e:
            raise Exception(f"Failed to extract text from PDF: {str(e)}")
    
    def validate_extracted_text(self, text: str) -> bool:
        """
        Validate that extracted text is not empty and contains meaningful content.
        
        Args:
            text: The extracted text to validate
            
        Returns:
            bool: True if text is valid, False otherwise
        """
        if not text or not text.strip():
            return False
        
        # Check if text has at least some alphabetic characters
        alphabetic_chars = sum(1 for char in text if char.isalpha())
        return alphabetic_chars > 50  # At least 50 alphabetic characters
