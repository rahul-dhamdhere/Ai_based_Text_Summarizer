import PyPDF2
from docx import Document
import io
import pytesseract
from pdf2image import convert_from_bytes
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def extract_text(file, filename):
   
    try:
        if filename.endswith('.pdf'):
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
            text = ''
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                text += page_text if page_text else ''
            if len(text.strip()) < 100:
                file.seek(0)
                images = convert_from_bytes(file.read())
                text = ''
                for image in images:
                    text += pytesseract.image_to_string(image, lang='eng') + '\n'
        elif filename.endswith('.docx'):
            doc = Document(io.BytesIO(file.read()))
            text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
        else:
            logging.error("Unsupported file format")
            return "Unsupported file format"
        logging.info("Text extracted successfully")
        return text
    except Exception as e:
        logging.error(f"Error extracting text: {str(e)}")
        return f"Error: {str(e)}"