"""
Enhanced Text Extractor Module for FastAPI
Handles extracting text content from PDF files with improved performance,
table extraction, and proper CID font handling for scripts like Malayalam.
"""
import pdfplumber
import pymupdf  # PyMuPDF (fitz)
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
import concurrent.futures
from pathlib import Path
import logging
import re
import unicodedata

class TextExtractor:
    """FastAPI-compatible text extractor with enhanced capabilities."""
   
    def __init__(self, max_workers: int = 4, backend: str = "auto"):
        """
        Initialize the enhanced text extractor.
        
        Args:
            max_workers: Number of worker threads for parallel processing
            backend: Extraction backend ('pdfplumber', 'pymupdf', 'auto')
        """
        self.max_workers = max_workers
        self.backend = backend
        self.logger = logging.getLogger(__name__)
        
    def extract_text_from_pdf(self, pdf_path: str, extract_tables: bool = True, 
                             handle_cid: bool = True) -> str:
        """
        Extract text from PDF file with tables in correct order.
       
        Args:
            pdf_path: Path to the PDF file
            extract_tables: Whether to extract and format tables inline
            handle_cid: Whether to handle CID font mapping issues
           
        Returns:
            str: Complete text content with tables in correct order
           
        Raises:
            Exception: If text extraction fails
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        self.logger.info(f"📖 Extracting text from PDF: {pdf_path.name}")
        
        # Choose extraction method based on backend
        if self.backend == "pymupdf" or (self.backend == "auto" and handle_cid):
            return self._extract_with_pymupdf(pdf_path, extract_tables, handle_cid)
        else:
            return self._extract_with_pdfplumber(pdf_path, extract_tables)
    
    def _extract_with_pymupdf(self, pdf_path: Path, extract_tables: bool, 
                             handle_cid: bool) -> str:
        """Extract using PyMuPDF with proper text flow handling."""
        
        def extract_page_content(page_data: Tuple[int, pymupdf.Page]) -> Dict:
            page_num, page = page_data
            result = {
                'page_num': page_num + 1,
                'content_blocks': []  # Will store text and table blocks in order
            }
            
            try:
                # Get all content blocks (text and potential table areas)
                text_blocks = []
                table_blocks = []
                
                # Extract text with proper formatting
                if handle_cid:
                    text_blocks = self._extract_text_blocks_with_proper_spacing(page)
                else:
                    # Simple text extraction
                    text = page.get_text()
                    if text.strip():
                        text_blocks.append({
                            'type': 'text',
                            'content': text,
                            'bbox': page.rect
                        })
                
                # Extract tables if requested
                if extract_tables:
                    tables = page.find_tables()
                    for i, table in enumerate(tables):
                        try:
                            table_data = table.extract()
                            if table_data and any(any(cell for cell in row if cell) for row in table_data):
                                formatted_table = self._format_table_as_text(table_data, page_num + 1, i + 1)
                                table_blocks.append({
                                    'type': 'table',
                                    'content': formatted_table,
                                    'bbox': table.bbox
                                })
                        except Exception as e:
                            self.logger.warning(f"Failed to extract table on page {page_num + 1}: {e}")
                
                # Combine and sort blocks by position
                all_blocks = text_blocks + table_blocks
                # Sort by y-coordinate (top to bottom)
                all_blocks.sort(key=lambda x: -x['bbox'][1])  # Negative for top-to-bottom
                
                result['content_blocks'] = all_blocks
                
            except Exception as e:
                self.logger.error(f"Error processing page {page_num + 1}: {e}")
                result['content_blocks'] = [{
                    'type': 'text',
                    'content': f"[Error extracting page {page_num + 1}]",
                    'bbox': (0, 0, 0, 0)
                }]
            
            return result
        
        try:
            doc = pymupdf.open(pdf_path)
            
            # Process pages in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                page_data = [(i, doc[i]) for i in range(len(doc))]
                page_results = list(executor.map(extract_page_content, page_data))
            
            # Compile final text with proper order
            full_text = ""
            
            for result in page_results:
                page_content = f"\n--- Page {result['page_num']} ---\n"
                
                # Add content blocks in order
                for block in result['content_blocks']:
                    if block['type'] == 'text' and block['content'].strip():
                        page_content += block['content'] + "\n"
                    elif block['type'] == 'table':
                        page_content += "\n" + block['content'] + "\n"
                
                full_text += page_content
            
            doc.close()
            
            # Clean up the final text
            full_text = self._clean_final_text(full_text)
            
            self.logger.info(f"✅ Extracted {len(full_text)} characters")
            return full_text
            
        except Exception as e:
            raise Exception(f"Failed to extract text with PyMuPDF: {str(e)}")
    
    def _extract_text_blocks_with_proper_spacing(self, page: pymupdf.Page) -> List[Dict]:
        """Extract text blocks with proper word spacing and line handling."""
        text_blocks = []
        
        # Get text as dictionary with detailed formatting info
        text_dict = page.get_text("dict")
        
        full_text = ""
        current_line_y = None
        current_line_text = ""
        
        for block in text_dict.get("blocks", []):
            if "lines" not in block:  # Skip non-text blocks
                continue
                
            block_text = ""
            
            for line in block["lines"]:
                line_bbox = line["bbox"]
                line_y = line_bbox[1]  # y-coordinate
                
                # Check if this is a new line (significant y-coordinate change)
                if current_line_y is not None and abs(line_y - current_line_y) > 3:
                    # Process the completed line
                    if current_line_text.strip():
                        processed_line = self._process_line_with_proper_spacing(current_line_text)
                        block_text += processed_line + "\n"
                    current_line_text = ""
                
                current_line_y = line_y
                
                # Extract text from spans with proper spacing
                line_text = ""
                prev_span_end = None
                
                for span in line["spans"]:
                    span_text = span.get("text", "")
                    span_bbox = span["bbox"]
                    
                    # Handle CID mapping issues
                    if self._has_cid_issues(span_text):
                        span_text = self._resolve_cid_text_advanced(span_text, span)
                    
                    # Add spacing between spans if there's a gap
                    if prev_span_end is not None:
                        gap = span_bbox[0] - prev_span_end
                        if gap > 5:  # Significant gap, add space
                            line_text += " "
                    
                    line_text += span_text
                    prev_span_end = span_bbox[2]  # Right edge of span
                
                current_line_text += line_text + " "
            
            # Process the last line
            if current_line_text.strip():
                processed_line = self._process_line_with_proper_spacing(current_line_text)
                block_text += processed_line + "\n"
                current_line_text = ""
            
            if block_text.strip():
                text_blocks.append({
                    'type': 'text',
                    'content': block_text,
                    'bbox': block["bbox"]
                })
        
        return text_blocks
    
    def _process_line_with_proper_spacing(self, line_text: str) -> str:
        """Process a line to ensure proper word spacing for Malayalam/complex scripts."""
        # Remove excessive spaces and newlines within the line
        line_text = re.sub(r'\s+', ' ', line_text.strip())
        
        # For Malayalam and similar scripts, ensure proper word boundaries
        # This is a basic implementation - you might need more sophisticated rules
        processed = ""
        words = line_text.split()
        
        for i, word in enumerate(words):
            # Clean individual words
            word = word.strip()
            if not word:
                continue
                
            # Add the word
            processed += word
            
            # Add space between words, but be careful with Malayalam conjuncts
            if i < len(words) - 1:
                next_word = words[i + 1].strip()
                if next_word and not self._is_conjunct_continuation(word, next_word):
                    processed += " "
        
        return processed
    
    def _is_conjunct_continuation(self, current_word: str, next_word: str) -> bool:
        """Check if the next word is a continuation of a Malayalam conjunct."""
        # Basic check for Malayalam conjuncts and joiners
        if not current_word or not next_word:
            return False
            
        # Check for Malayalam zero-width joiner or similar cases
        malayalam_range = range(0x0D00, 0x0D80)
        
        # If both words contain Malayalam characters and current word ends with certain characters
        current_has_malayalam = any(ord(c) in malayalam_range for c in current_word)
        next_has_malayalam = any(ord(c) in malayalam_range for c in next_word)
        
        if current_has_malayalam and next_has_malayalam:
            # Check for specific Malayalam joining patterns
            if current_word.endswith(('്', '്‍')) or next_word.startswith(('്', '്‍')):
                return True
        
        return False
    
    def _has_cid_issues(self, text: str) -> bool:
        """Check if text has CID mapping issues."""
        return bool(re.search(r'\(cid:\d+\)|cid-\d+', text, re.IGNORECASE))
    
    def _resolve_cid_text_advanced(self, text: str, span: Dict) -> str:
        """Advanced CID resolution with better character recovery."""
        if not self._has_cid_issues(text):
            return text
        
        # Try to extract readable parts
        processed = text
        
        # Handle different CID patterns
        processed = re.sub(r'\(cid:\d+\)', '', processed, flags=re.IGNORECASE)
        processed = re.sub(r'cid-\d+', '', processed, flags=re.IGNORECASE)
        
        # Clean up multiple spaces
        processed = re.sub(r'\s+', ' ', processed).strip()
        
        return processed
    
    def _format_table_as_text(self, table_data: List[List], page_num: int, table_num: int) -> str:
        """Format table data as readable text."""
        if not table_data:
            return ""
        
        # Convert to DataFrame for better formatting
        try:
            # Handle cases where first row might not be headers
            df = pd.DataFrame(table_data)
            
            # Clean the data
            df = df.fillna('')
            df = df.astype(str)
            
            # Format as text
            formatted = f"\n{'='*60}\n"
            formatted += f"TABLE {table_num} (Page {page_num})\n"
            formatted += f"{'='*60}\n"
            
            # Use pandas to_string for clean formatting
            formatted += df.to_string(index=False, header=False, max_colwidth=30)
            formatted += f"\n{'='*60}\n"
            
            return formatted
            
        except Exception as e:
            self.logger.warning(f"Failed to format table: {e}")
            return f"\n[TABLE {table_num} - Page {page_num}: Formatting Error]\n"
    
    def _extract_with_pdfplumber(self, pdf_path: Path, extract_tables: bool) -> str:
        """Extract using pdfplumber with inline table handling."""
        
        def extract_page_content(page_data: Tuple[int, object]) -> str:
            page_num, page = page_data
            page_content = f"\n--- Page {page_num + 1} ---\n"
            
            try:
                # Extract main text
                text = page.extract_text() or ""
                
                if extract_tables:
                    # Get table locations to insert them in correct positions
                    tables = page.extract_tables()
                    
                    if tables:
                        # For simplicity, append tables at the end of page text
                        page_content += text + "\n"
                        
                        for i, table in enumerate(tables):
                            if table:
                                formatted_table = self._format_table_as_text(table, page_num + 1, i + 1)
                                page_content += formatted_table + "\n"
                    else:
                        page_content += text + "\n"
                else:
                    page_content += text + "\n"
                
            except Exception as e:
                self.logger.error(f"Error processing page {page_num + 1}: {e}")
                page_content += f"[Error extracting page {page_num + 1}]\n"
            
            return page_content
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                # Process pages in parallel
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    page_data = [(i, pdf.pages[i]) for i in range(len(pdf.pages))]
                    page_results = list(executor.map(extract_page_content, page_data))
            
            # Combine all pages
            full_text = "".join(page_results)
            full_text = self._clean_final_text(full_text)
            
            self.logger.info(f"✅ Extracted {len(full_text)} characters")
            return full_text
            
        except Exception as e:
            raise Exception(f"Failed to extract text with pdfplumber: {str(e)}")
    
    def _clean_final_text(self, text: str) -> str:
        """Final cleanup of extracted text."""
        # Remove excessive line breaks but preserve paragraph structure
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Max 2 consecutive newlines
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs to single space
        text = re.sub(r'[ \t]*\n[ \t]*', '\n', text)  # Clean spaces around newlines
        
        # Remove trailing whitespace from each line
        lines = text.split('\n')
        cleaned_lines = [line.rstrip() for line in lines]
        text = '\n'.join(cleaned_lines)
        
        return text.strip()
    
    def validate_extracted_text(self, text: str, min_chars: int = 50) -> Tuple[bool, Dict]:
        """
        Validate extracted text with detailed feedback.
       
        Args:
            text: The extracted text to validate
            min_chars: Minimum number of alphabetic characters required
           
        Returns:
            Tuple of (is_valid, validation_details)
        """
        details = {
            'total_chars': len(text) if text else 0,
            'alphabetic_chars': 0,
            'has_content': bool(text and text.strip()),
            'cid_issues': 0,
            'validation_passed': False,
            'line_breaks_ratio': 0
        }
        
        if not text or not text.strip():
            return False, details
        
        # Count different character types
        details['alphabetic_chars'] = sum(1 for char in text if char.isalpha())
        
        # Count CID issues
        cid_matches = re.findall(r'\(cid:\d+\)|cid-\d+|\[\?\]', text, re.IGNORECASE)
        details['cid_issues'] = len(cid_matches)
        
        # Check line break ratio (to detect excessive line breaking)
        newline_count = text.count('\n')
        if details['total_chars'] > 0:
            details['line_breaks_ratio'] = newline_count / details['total_chars']
        
        # Validation logic
        details['validation_passed'] = (
            details['alphabetic_chars'] >= min_chars and 
            details['line_breaks_ratio'] < 0.1  # Less than 10% line breaks
        )
        
        return details['validation_passed'], details

# Simple synchronous usage (without FastAPI)
def extract_pdf_simple(pdf_path: str) -> str:
    """Simple extraction function for direct use."""
    extractor = TextExtractor()
    return extractor.extract_text_from_pdf(pdf_path)