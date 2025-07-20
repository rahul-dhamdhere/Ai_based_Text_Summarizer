import nltk
import numpy as np
from nltk.tokenize import sent_tokenize
import logging
import sys
import ollama
from sklearn.metrics.pairwise import cosine_similarity

# Download required NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Preprocess text (basic cleaning)
def preprocess_text(text):
    
    text = ' '.join(text.strip().split())
    return text

# Extractive summarization with nomic-embed-text
def summarize_with_nomic(text, summary_ratio=0.65, min_sentences=3):
    
    try:
        # Preprocess and tokenize
        text = preprocess_text(text)
        sentences = sent_tokenize(text)
        if len(sentences) <= min_sentences:
            logging.info("Text too short for summarization, returning original")
            return text

        # Generate document embedding
        full_response = ollama.embeddings(model='nomic-embed-text', prompt=text)
        doc_embedding = np.array(full_response['embedding']).reshape(1, -1)

        # Generate sentence embeddings
        sentence_embeddings = []
        for sent in sentences:
            sent_resp = ollama.embeddings(model='nomic-embed-text', prompt=sent)
            sentence_embeddings.append(sent_resp['embedding'])
        sentence_embeddings = np.array(sentence_embeddings)

        # Compute cosine similarity
        sims = cosine_similarity(sentence_embeddings, doc_embedding).flatten()
        num_sentences = max(min_sentences, int(len(sentences) * summary_ratio))
        ranked_indices = np.argsort(sims)[::-1][:num_sentences]
        selected_indices = sorted(ranked_indices)

        # Generate summary
        summary = ' '.join(sentences[i] for i in selected_indices)
        logging.info(f"Extractive summary generated: {summary}")
        return summary
    except Exception as e:
        logging.error(f"Error generating nomic summary: {str(e)}")
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

    # Summarize text
    summary = summarize_with_nomic(text, summary_ratio=0.4)
    print("\nExtractive Summary:")
    print(summary if summary else "Error: Could not generate summary.")

if __name__ == "__main__":
    main()