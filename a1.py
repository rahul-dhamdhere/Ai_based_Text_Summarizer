import re
import nltk
import numpy as np
import networkx as nx
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
import sys

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Preprocess text (with audio-specific cleaning)
def preprocess_text(text, is_audio=False):
   
    text = re.sub(r'\s+', ' ', text.strip())
    if is_audio:
        text = re.sub(r'\b(um|uh|like)\b', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s+', ' ', text.strip())
    return text

# Preprocess sentence for similarity matrix
def preprocess_sentence(sentence, stop_words):
    
    words = [word.lower() for word in word_tokenize(sentence) if word.isalnum()]
    return [word for word in words if word not in stop_words]

# Build similarity matrix using TF-IDF
def build_similarity_matrix(sentences, stop_words):
    
    try:
        vectorizer = TfidfVectorizer(stop_words=list(stop_words))
        tfidf_matrix = vectorizer.fit_transform(sentences)
        return cosine_similarity(tfidf_matrix)
    except Exception as e:
        logging.error(f"Error building similarity matrix: {str(e)}")
        return np.zeros((len(sentences), len(sentences)))

# Extract dynamic keywords using TF-IDF
def get_dynamic_keywords(text, top_n=5):
    
    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([text])
        feature_names = vectorizer.get_feature_names_out()
        scores = tfidf_matrix.toarray()[0]
        return [feature_names[i] for i in scores.argsort()[-top_n:]]
    except Exception as e:
        logging.error(f"Error extracting keywords: {str(e)}")
        return []

# Enhanced TextRank summarization
def summarize_text_rank_mixed(text, ratio=0.66, alpha=0.7, boost_factor=0.2, is_audio=False):
    
    try:
        # Preprocess text
        text = preprocess_text(text, is_audio=is_audio)
        sentences = sent_tokenize(text)
        if not sentences:
            logging.warning("No sentences detected in text")
            return ""

        # Initialize stop words and keywords
        stop_words = set(stopwords.words('english'))
        boost_keywords = get_dynamic_keywords(text, top_n=5)

        # Build similarity matrix
        similarity_matrix = build_similarity_matrix(sentences, stop_words)
        if similarity_matrix.size == 0:
            logging.warning("Empty similarity matrix, returning first sentence")
            return sentences[0] if sentences else ""

        # Build graph and compute PageRank
        sentence_graph = nx.from_numpy_array(similarity_matrix)
        try:
            pagerank_scores = nx.pagerank(sentence_graph, max_iter=100, tol=1e-4)
        except nx.PowerIterationFailedConvergence:
            logging.error("PageRank convergence failed; using uniform scores")
            pagerank_scores = {i: 1/len(sentences) for i in range(len(sentences))}

        # Normalize PageRank scores
        max_pr = max(pagerank_scores.values()) if pagerank_scores else 1
        pagerank_scores = {k: v / max_pr for k, v in pagerank_scores.items()}

        # Compute combined scores
        combined_scores = {}
        for i, sentence in enumerate(sentences):
            position_score = 1 / (1 + np.log1p(i))  # Favor earlier sentences
            pr_score = pagerank_scores.get(i, 0)
            keyword_bonus = boost_factor if any(kw.lower() in sentence.lower() for kw in boost_keywords) else 0
            combined_scores[i] = alpha * pr_score + (1 - alpha) * position_score + keyword_bonus

        # Select sentences
        ranked_indices = sorted(combined_scores, key=combined_scores.get, reverse=True)
        num_sentences = max(1, int(len(sentences) * ratio))
        forced_indices = {0}  # Always include first sentence
        selected_indices = list(forced_indices)
        for idx in ranked_indices:
            if idx not in selected_indices and len(selected_indices) < num_sentences:
                selected_indices.append(idx)

        # Generate summary
        summary = ' '.join(sentences[i] for i in sorted(selected_indices))
        logging.info(f"Extractive summary generated: {summary}")
        return summary
    except Exception as e:
        logging.error(f"Error in TextRank summarization: {str(e)}")
        return f"Error: {str(e)}"

# Main function for terminal input/output
def main():
    
    print("Enter text to summarize (press Enter twice to finish):")
    lines = []
    while True:
        try:
            line = input()
            if line == "":  # Blank line
                if lines and lines[-1] == "":  # Two blank lines
                    break
                lines.append("")
            else:
                lines.append(line)
        except EOFError:  # Handle Ctrl+D
            break
        except KeyboardInterrupt:  # Handle Ctrl+C
            print("\nExiting...")
            sys.exit(0)

    text = " ".join(lines).strip()
    if not text:
        print("Error: No text provided.")
        return

    # Summarize text (treat as audio for preprocessing)
    summary = summarize_text_rank_mixed(text, ratio=0.4, is_audio=True)
    print("\nExtractive Summary:")
    print(summary if summary else "Error: Could not generate summary.")

if __name__ == "__main__":
    main()