import re
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def preprocess_text(text, is_audio=False):
    
    text = re.sub(r'\s+', ' ', text.strip())
    if is_audio:
        text = re.sub(r'\b(um|uh|like|you know)\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s+', ' ', text.strip())
        text = re.sub(r'([a-zA-Z])\s*$', r'\1.', text)
    return text