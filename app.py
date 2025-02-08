from flask import Flask, render_template, request, jsonify
import nltk
import numpy as np
import networkx as nx
import ollama
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import os

# Initialize Flask app
app = Flask(__name__)

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Copy the text processing functions from backend.py
def sentence_similarity(sent1, sent2, stop_words=None):
    if stop_words is None:
        stop_words = set(stopwords.words('english'))
    
    sent1_words = [word.lower() for word in word_tokenize(sent1) if word.isalnum()]
    sent2_words = [word.lower() for word in word_tokenize(sent2) if word.isalnum()]
    
    sent1_words = [word for word in sent1_words if word not in stop_words]
    sent2_words = [word for word in sent2_words if word not in stop_words]
    
    all_words = list(set(sent1_words + sent2_words))
    
    vector1 = [sent1_words.count(word) for word in all_words]
    vector2 = [sent2_words.count(word) for word in all_words]
    
    numerator = np.dot(vector1, vector2)
    denominator = np.sqrt(np.sum(np.square(vector1))) * np.sqrt(np.sum(np.square(vector2)))
    return numerator / denominator if denominator != 0 else 0.0

def build_similarity_matrix(sentences, stop_words=None):
    n = len(sentences)
    similarity_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                similarity_matrix[i][j] = sentence_similarity(sentences[i], sentences[j], stop_words)
    
    return similarity_matrix

def summarize_text_rank_mixed(text, ratio=0.4, alpha=0.7, boost_keywords=None, boost_factor=0.2):
    if boost_keywords is None:
        boost_keywords = {'goal', 'understanding', 'extract', 'organize', 'historical', 'Turing'}
    
    sentences = sent_tokenize(text)
    if not sentences:
        return ""
    
    stop_words = set(stopwords.words('english'))
    similarity_matrix = build_similarity_matrix(sentences, stop_words)
    sentence_similarity_graph = nx.from_numpy_array(similarity_matrix)
    pagerank_scores = nx.pagerank(sentence_similarity_graph)
    
    combined_scores = {}
    n = len(sentences)
    for i, s in enumerate(sentences):
        position_score = (n - i) / n
        pr_score = pagerank_scores[i]
        keyword_bonus = boost_factor if any(kw.lower() in s.lower() for kw in boost_keywords) else 0
        combined_scores[i] = alpha * pr_score + (1 - alpha) * position_score + keyword_bonus
    
    ranked_indices = sorted(combined_scores, key=combined_scores.get, reverse=True)
    num_sentences = max(1, int(len(sentences) * ratio))
    
    forced_indices = {0}
    selected_indices = list(forced_indices)
    for idx in ranked_indices:
        if idx not in selected_indices and len(selected_indices) < num_sentences:
            selected_indices.append(idx)
    
    selected_indices = sorted(selected_indices)
    return ' '.join([sentences[i] for i in selected_indices])

def generate_abstractive_content(text, summary_ratio=0.4):
    extractive_summary = summarize_text_rank_mixed(text, ratio=summary_ratio)
    
    prompt = f"""Transform this technical summary into two formats:

1. A concise, hierarchical report in note-making format using bullet points
2. Mermaid code for a memory map (mind map) showing key concepts and relationships

Summary:
{extractive_summary}

Format exactly like this:
[Report in bullet point format...]

```mermaid
[Valid Mermaid diagram code...]
```"""
    
    try:
        response = ollama.generate(
            model='deepseek-r1:1.5b',
            prompt=prompt,
            stream=False
        )['response']
        
        if '```mermaid' in response:
            report_part, mermaid_part = response.split('```mermaid', 1)
            mermaid_code = mermaid_part.split('```')[0].strip()
            print(f"extractive_summary : {extractive_summary}")
            print(f"abstractive_report: {report_part}")
            return {
                'extractive_summary': extractive_summary,
                'abstractive_report': report_part.strip(),
                'mermaid_code': f"```mermaid\n{mermaid_code}\n```"
            }
        return {
            'extractive_summary': extractive_summary,
            'abstractive_report': response,
            'mermaid_code': ""
        }
    except Exception as e:
        return {
            'extractive_summary': extractive_summary,
            'abstractive_report': f"Error generating content: {str(e)}",
            'mermaid_code': ""
        }

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
    return jsonify(result)

if __name__ == '__main__':
    # Ensure templates directory exists
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    # Move the HTML file to templates directory
    with open('templates/index.html', 'w') as f:
        with open('index.html', 'r') as source:
            f.write(source.read())
    
    app.run(debug=True)