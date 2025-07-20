# AI Summarizer - Academic Project

## Overview

**AI Summarizer** is an academic project designed to provide advanced text and audio summarization using state-of-the-art Natural Language Processing (NLP) and Large Language Models (LLMs). The application allows users to summarize documents (PDF, DOCX), transcribe and summarize audio, and interact with an AI-powered chatbot for follow-up questions and clarifications. The project demonstrates integration of extractive and abstractive summarization techniques, audio transcription, file text extraction, and conversational AI.

---

## Features

- **Text Summarization**: Supports both extractive and abstractive summaries for user-provided text.
- **Audio Transcription & Summarization**: Record or upload audio, transcribe it using Whisper, and generate summaries.
- **File Upload**: Extracts text from PDF and DOCX files, including OCR for scanned PDFs.
- **Chatbot**: Conversational assistant for follow-up questions, clarifications, and summary modifications.
- **Chat History**: Stores and retrieves chat sessions for each user.
- **Export**: Download summary responses as PDF.
- **Modern UI**: Responsive, accessible web interface built with HTML, CSS, and JavaScript.
- **Academic Documentation**: Well-commented code and modular design for educational purposes.

---

## Technologies Used

- **Python** (Flask backend)
- **JavaScript** (Frontend interactivity)
- **HTML/CSS** (User interface)
- **Ollama** (Local LLM inference for summarization and chat)
- **Whisper** (Audio transcription)
- **NLTK, Scikit-learn, NetworkX** (Extractive summarization algorithms)
- **PyPDF2, python-docx, pytesseract, pdf2image** (File text extraction)
- **SQLite** (Chat history storage)

---

## Project Structure

```
main-project/
│
├── app.py                   # Flask application entry point
├── chatbot.py               # Chatbot logic and chat history management
├── extractive_summary.py    # Extractive summarization (nomic-embed-text)
├── abstractive_summary.py   # Abstractive summarization (LLM-based)
├── transcription.py         # Audio transcription using Whisper
├── file_extraction.py       # PDF/DOCX text extraction and OCR
├── utils.py                 # Text preprocessing utilities
├── a1.py                    # Alternative extractive summarization (TextRank)
├── templates/
│   └── index.html           # Main web interface
├── chat_history.db          # SQLite database for chat storage
└── README.md                # Project documentation
```

---

## Setup Instructions

### 1. Prerequisites

- Python 3.8+
- [Ollama](https://ollama.com/) installed and running locally (for LLM inference)
- Whisper (install via `pip install openai-whisper`)
- Tesseract OCR (for scanned PDF extraction)
- Required Python packages (see below)

### 2. Install Dependencies

Open terminal in the project directory and run:

```bash
pip install flask openai-whisper nltk scikit-learn networkx PyPDF2 python-docx pytesseract pdf2image ollama
```

Install Tesseract OCR (Windows):

- Download from: https://github.com/tesseract-ocr/tesseract
- Add Tesseract to your system PATH.

### 3. Download NLTK Data

The first run will download required NLTK resources. If you encounter errors, run:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

### 4. Start Ollama

Download and run Ollama from [https://ollama.com/download](https://ollama.com/download).

Pull required models:

```bash
ollama pull deepseek-r1:8b
ollama pull deepseek-r1:1.5b
ollama pull nomic-embed-text
```

### 5. Run the Application

Start the Flask server:

```bash
python app.py
```

Open your browser and go to [http://localhost:5000](http://localhost:5000).

---

## Usage Guide

1. **Summarize Text**: Paste or type text in the input area and click "Generate Summary".
2. **Upload File**: Click "Upload File" to select a PDF or DOCX. The text will be extracted and summarized.
3. **Audio Summarization**: Click "Start" to record audio, then "Stop" to transcribe and summarize.
4. **Chatbot**: Use the chat interface to ask questions about the summary, request clarifications, or modify summaries.
5. **Chat History**: View, reload, or delete previous chat sessions.
6. **Export**: Download summary responses as PDF for academic records or sharing.

---

## Academic Notes

- **Extractive Summarization**: Uses sentence embeddings and cosine similarity to select key sentences.
- **Abstractive Summarization**: Refines extractive summaries using LLMs for coherent, human-like summaries.
- **Audio Transcription**: Employs Whisper for high-accuracy speech-to-text.
- **File Extraction**: Handles both digital and scanned documents using OCR.
- **Chatbot**: Context-aware, leverages chat history and summaries for informed responses.
- **Data Privacy**: No personal data is stored; chat sessions are keyed by user IP for demonstration.

---

## Limitations & Future Work

- **Local Models**: Requires significant system resources for LLMs and Whisper.
- **Scalability**: Designed for academic/demo use, not production scale.
- **Security**: No authentication; for academic demonstration only.
- **Model Customization**: Can be extended with other LLMs or summarization techniques.

---

## References

- [Ollama](https://ollama.com/)
- [OpenAI Whisper](https://github.com/openai/whisper)
- [NLTK](https://www.nltk.org/)
- [Scikit-learn](https://scikit-learn.org/)
- [NetworkX](https://networkx.org/)
- [PyPDF2](https://pypdf2.readthedocs.io/)
- [python-docx](https://python-docx.readthedocs.io/)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)

---

## License

This project is for academic use only. Please cite appropriately if used in research or coursework.

---

## Contact

For questions or academic collaboration, contact the project author or supervisor.
