import re
import logging
import ollama
from extractive_summary import summarize_with_nomic
from utils import preprocess_text

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_abstractive_content(text, summary_ratio=0.5, is_audio=False, retries=2):
    try:
        text = preprocess_text(text, is_audio=is_audio)
        if not text:
            logging.error("Input text is empty after preprocessing.")
            return {'extractive_summary': '', 'abstractive_report': 'Error: Input text is empty after preprocessing.'}

        extractive_summary = summarize_with_nomic(text, summary_ratio=summary_ratio)
        if not extractive_summary or 'error' in extractive_summary.lower():
            logging.error(f"Extractive summarization failed: {extractive_summary}")
            return {'extractive_summary': '', 'abstractive_report': f"Error: Extractive summarization failed. {extractive_summary}"}

        # Truncate extractive summary if too long for the model (adjust depending on model token limits)
        max_extract_length = 3000  # approx characters
        if len(extractive_summary) > max_extract_length:
            logging.warning(f"Extractive summary too long ({len(extractive_summary)} characters). Truncating.")
            extractive_summary = extractive_summary[:max_extract_length] + "..."

        # Structured prompt
        prompt = f"""
            You are an expert assistant for summarization tasks.

            Your task:
            - Refine the provided extractive summary into a clear, concise, professional summary.
            - Maintain key facts and important points.
            - Keep it structured, logical, and readable.

            Extractive Summary:
            \"\"\"
            {extractive_summary}
            \"\"\"

            Please generate the improved summary below:
            """

        logging.debug(f"Prompt to model: {prompt[:500]}...")  # Log only part of the prompt

        attempt = 0
        abstractive_summary = ''
        while attempt <= retries:
            try:
                response = ollama.generate(
                    model='deepseek-r1:8b',
                    prompt=prompt,
                    stream=False
                )
                abstractive_summary = response.get('response', '').strip()
                if abstractive_summary:
                    break
            except Exception as e:
                logging.error(f"Attempt {attempt+1} failed: {str(e)}")
            attempt += 1

        if not abstractive_summary:
            return {
                'extractive_summary': extractive_summary,
                'abstractive_report': 'Error: Failed to generate abstractive summary after retries.'
            }

        # Clean up extra tags like <think> if present
        abstractive_summary = re.sub(r'<think>.*?</think>', '', abstractive_summary, flags=re.DOTALL).strip()

        logging.info("Abstractive summary generated successfully.")
        return {
            'extractive_summary': extractive_summary,
            'abstractive_report': abstractive_summary
        }

    except Exception as e:
        logging.error(f"Error in abstractive summary generation: {str(e)}")
        return {'extractive_summary': '', 'abstractive_report': f"Error generating content: {str(e)}"}
