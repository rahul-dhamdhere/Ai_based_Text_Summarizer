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
    
    prompt = f"""Transform this extractive summary into a detailed summary. 
    - Rephrase sentences naturally.
    - Use clear, concise language.
    - Preserve key facts and named entities.

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
    summary_length = data.get('summary_length', 'medium')
    
    # Adjust ratio based on summary length
    ratio_map = {
        'short': 0.2,
        'medium': 0.4,
        'detailed': 0.6
    }
    ratio = ratio_map.get(summary_length, 0.4)
    
    # Generate summary
    result = generate_abstractive_content(text, summary_ratio=ratio)
    
    # Store the summary (using a simple user_id for demo; replace with session in real app)
    user_id = request.remote_addr  # Use IP as a basic identifier
    summaries[user_id] = result
    
    return jsonify(result)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_id = request.remote_addr
    user_message = data.get('message', '').lower().strip()
    
    # Check cache first
    cache_key = f"{user_id}:{user_message}"
    if cache_key in chat_cache:
        return jsonify({'response': chat_cache[cache_key]})
    
    # Get previous summary
    prev_summary = summaries.get(user_id, {'extractive_summary': '', 'abstractive_report': ''})
    extractive = prev_summary['extractive_summary']
    abstractive = prev_summary['abstractive_report']
    
    # Handle specific requests
    if "shorter" in user_message:
        # Instruct LLM to shorten the existing abstractive summary
        prompt = f"""
        You are a helpful assistant for a text summarization tool. The user’s current summary is:
        - Abstractive Summary: {abstractive}
        
        The user asks: "{user_message}"
        Shorten the existing abstractive summary even further. Keep key facts and entities intact.
        Respond only with the shortened summary.
        """
    elif "info" in user_message or "tell me about" in user_message:
        # Extract topic from message (simple keyword approach)
        topic = user_message.replace("info about", "").replace("tell me about", "").strip()
        prompt = f"""
        You are a helpful assistant for a text summarization tool. The user’s summaries are:
        - Extractive Summary: {extractive}
        - Abstractive Summary: {abstractive}
        
        The user asks: "{user_message}"
        Provide concise, relevant information about "{topic}". Use the summaries if relevant, otherwise provide general knowledge.
        """
    else:
        # General handling with context
        prompt = f"""
        You are a helpful assistant for a text summarization tool. The user’s summaries are:
        - Extractive Summary: {extractive}
        - Abstractive Summary: {abstractive}
        
        The user asks: "{user_message}"
        Respond naturally, modifying the summary if requested or providing useful info based on the summaries.
        """
    
    try:
        response = ollama.generate(
            model='deepseek-r1:1.5b',
            prompt=prompt,
            stream=False
        )
        chat_response = response.get('response', 'Error: No response from model.')
        
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
