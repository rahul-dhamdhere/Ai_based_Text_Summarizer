import nltk
import numpy as np
import networkx as nx
import ollama
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from flask import Flask, render_template, request, jsonify
import PyPDF2
from docx import Document
import io
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
import tempfile

# Initialize Flask app
app = Flask(__name__)

# Download required NLTK resources
nltk.download('punkt')  # Fix here
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')  # Extra fix if needed
nltk.download('punkt_tab')

from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to preprocess and tokenize sentences, removing stopwords
def preprocess_sentence(sentence, stop_words):
    words = [word.lower() for word in word_tokenize(sentence) if word.isalnum()]
    return [word for word in words if word not in stop_words]

# Optimized sentence similarity using TF-IDF
def sentence_similarity(sent1, sent2, stop_words):
    sent1_words = preprocess_sentence(sent1, stop_words)
    sent2_words = preprocess_sentence(sent2, stop_words)

    if not sent1_words or not sent2_words:
        return 0.0

    all_words = list(set(sent1_words + sent2_words))
    vector1 = np.array([sent1_words.count(word) for word in all_words])
    vector2 = np.array([sent2_words.count(word) for word in all_words])

    return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2) + 1e-10)

# Build similarity matrix with TF-IDF enhancement
def build_similarity_matrix(sentences, stop_words):
    vectorizer = TfidfVectorizer(stop_words=list(stop_words))
    tfidf_matrix = vectorizer.fit_transform(sentences)
    return cosine_similarity(tfidf_matrix)

# Enhanced TextRank with adaptive keyword boosting and position weighting
def summarize_text_rank_mixed(text, ratio=0.4, alpha=0.7, boost_keywords=None, boost_factor=0.2):
    if boost_keywords is None:
        boost_keywords = {'goal', 'understanding', 'extract', 'organize', 'historical', 'Turing'}
    
    sentences = sent_tokenize(text)
    if not sentences:
        return ""

    stop_words = set(stopwords.words('english'))
    similarity_matrix = build_similarity_matrix(sentences, stop_words)

    # Convert similarity matrix to NumPy array before graph construction
    similarity_matrix = np.array(similarity_matrix)  # Fix here
    sentence_graph = nx.from_numpy_array(similarity_matrix)

    # Compute PageRank
    pagerank_scores = nx.pagerank(sentence_graph, max_iter=100, tol=1e-4)

    # Normalize PageRank scores
    max_pr = max(pagerank_scores.values()) if pagerank_scores else 1
    pagerank_scores = {k: v / max_pr for k, v in pagerank_scores.items()}

    combined_scores = defaultdict(float)
    n = len(sentences)

    for i, sentence in enumerate(sentences):
        position_score = (n - i) / n  # Higher weight for earlier sentences
        pr_score = pagerank_scores.get(i, 0)
        keyword_bonus = boost_factor if any(kw.lower() in sentence.lower() for kw in boost_keywords) else 0
        combined_scores[i] = alpha * pr_score + (1 - alpha) * position_score + keyword_bonus

    # Select top-ranked sentences
    ranked_indices = sorted(combined_scores, key=combined_scores.get, reverse=True)
    num_sentences = max(1, int(n * ratio))

    forced_indices = {0}
    selected_indices = list(forced_indices)
    for idx in ranked_indices:
        if idx not in selected_indices and len(selected_indices) < num_sentences:
            selected_indices.append(idx)

    return ' '.join(sentences[i] for i in sorted(selected_indices))

summaries = {}  # Store user summaries
chat_cache = {}  # Cache chat responses for optimization

def generate_abstractive_content(text, summary_ratio=0.4):
    extractive_summary = summarize_text_rank_mixed(text, ratio=summary_ratio)
    
    prompt = f"""
        You are a text summarization assistant. 
        Your task is to transform the following extractive summary into a clear, concise, and readable summary while preserving key facts. 
        - Do not add fictional or generic information.
        - Only include information from the extractive summary.
        - Use clear and professional language.
        - Keep it concise.

    Extractive Summary:
    {extractive_summary}
    
    Abstractive Summary:
    """
    print(extractive_summary)
    try:
        response = ollama.generate(
            model='deepseek-r1:1.5b',
            prompt=prompt,
            stream=False
        )
        
        # Ensure 'response' key exists in the output
        abstractive_summary = response.get('response', 'Error: No response received.')
        
        return {
            'extractive_summary': extractive_summary,
            'abstractive_report': abstractive_summary
        }
    except Exception as e:
        return {
            'extractive_summary': extractive_summary,
            'abstractive_report': f"Error generating content: {str(e)}"
        }
summaries = {}

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_text():
    data = request.get_json()
    text = data.get('text', '')

    # Generate summary
    result = generate_abstractive_content(text, summary_ratio=0.4)

    # Store the input text and summaries (using a simple user_id for demo; replace with session in real app)
    user_id = request.remote_addr  # Use IP as a basic identifier
    summaries[user_id] = {
        'input_text': text,
        'extractive_summary': result['extractive_summary'],
        'abstractive_report': result['abstractive_report']
    }

    return jsonify(result)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_id = request.remote_addr
    user_message = data.get('message', '').strip()  # Keep original case for natural feel

    # Check cache first
    cache_key = f"{user_id}:{user_message}"
    if cache_key in chat_cache:
        return jsonify({'response': chat_cache[cache_key]})

    # Get previous summary and input text
    prev_summary = summaries.get(user_id, {'extractive_summary': '', 'abstractive_report': ''})
    extractive = prev_summary['extractive_summary']
    abstractive = prev_summary['abstractive_report']
    input_text = prev_summary.get('input_text', '')

    # Single, flexible prompt
    prompt = f"""
    You are a helpful assistant for a text summarization tool. The user has provided the following input and generated summaries:
    - Original Input: {input_text}
    - Extractive Summary: {extractive}
    - Abstractive Summary genrated by llm using extracive summary: {abstractive}

    The user asks: "{user_message}"

    Interpret the user's request naturally and respond appropriately. Follow these guidelines:
    - If they ask to modify the summary (e.g., "make it shorter," "simplify it", "describe more"), adjust the abstractive summary accordingly and return only the modified version.
    - If they request information (e.g., "tell me about X," "what is Y"), provide concise, relevant info, using the input and summaries if applicable, or general knowledge if not.
    - For other queries, respond naturally, leveraging the input and summaries as context where relevant.
    - Keep responses clear, concise, and focused on the user's intent.
    """

    try:
        response = ollama.generate(
            model='deepseek-r1:1.5b',
            prompt=prompt,
            stream=False
        )
        chat_response = response.get('response', 'Error: No response from model.')
        # Clean <think> tags
        import re
        chat_response = re.sub(r'<think>.*?</think>', '', chat_response, flags=re.DOTALL)

        # Cache the response
        chat_cache[cache_key] = chat_response
    except Exception as e:
        chat_response = f"Error: {str(e)}"

    return jsonify({'response': chat_response})

@app.route('/extract-text', methods=['POST'])
def extract_text():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        # Handle PDF files
        if file.filename.endswith('.pdf'):
            # Try normal extraction first
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
            text = ''
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                text += page_text

            # If normal extraction yields very little text, try OCR
            if len(text.strip()) < 100:  # Arbitrary threshold
                file.seek(0)  # Reset file pointer
                # Convert PDF to images
                images = convert_from_bytes(file.read())
                text = ''
                for image in images:
                    # Perform OCR on each page
                    text += pytesseract.image_to_string(image, lang='eng') + '\n'
        
        # Handle DOCX files
        elif file.filename.endswith('.docx'):
            doc = Document(io.BytesIO(file.read()))
            text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
        
        else:
            return jsonify({'error': 'Unsupported file format'}), 400

        return jsonify({'text': text})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
