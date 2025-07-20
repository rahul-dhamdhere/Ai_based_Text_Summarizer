from flask import Flask, request, jsonify, render_template
import logging
import uuid
from transcription import transcribe_audio
from extractive_summary import summarize_with_nomic
from abstractive_summary import generate_abstractive_content
from chatbot import chat_with_model, get_chat_history, load_chat
from file_extraction import extract_text
from utils import preprocess_text
import sqlite3
# Setup Flask app
app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Global summaries storage
summaries = {}

@app.route('/')
def index():
    
    return render_template('index.html')

@app.route('/record-audio', methods=['POST'])
def record_audio():
    
    if 'audio' not in request.files:
        logging.error("No audio file provided")
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    if audio_file.filename == '':
        logging.error("No audio file selected")
        return jsonify({'error': 'No audio file selected'}), 400

    try:
        audio_data = audio_file.read()
        transcription = transcribe_audio(audio_data)
        if transcription.startswith('Transcription error'):
            logging.error(f"Transcription failed: {transcription}")
            return jsonify({'error': transcription}), 500

        user_id = request.remote_addr
        chat_id = str(uuid.uuid4())

        from chatbot import DATABASE
        with sqlite3.connect(DATABASE) as conn:
            cursor = conn.cursor()
            cursor.execute('INSERT INTO Chat (id, user_id) VALUES (?, ?)', (chat_id, user_id))
            cursor.execute('INSERT INTO Message (chat_id, role, content) VALUES (?, ?, ?)', (chat_id, 'user', f"Audio transcription: {transcription}"))
            conn.commit()
            logging.info(f"Audio transcription saved to database, chat_id: {chat_id}")

        return jsonify({'transcription': transcription, 'chat_id': chat_id})
    except Exception as e:
        logging.error(f"Error processing audio: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/transcribe', methods=['POST'])
def transcribe():
    
    if 'audio' not in request.files:
        logging.error("No audio file provided")
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    if audio_file.filename == '':
        logging.error("No audio file selected")
        return jsonify({'error': 'No audio file selected'}), 400

    try:
        audio_data = audio_file.read()
        transcription = transcribe_audio(audio_data)
        
        if transcription.startswith('Transcription error'):
            logging.error(f"Transcription failed: {transcription}")
            return jsonify({'error': transcription}), 500

        logging.info("Transcription successful")
        return jsonify({
            'text': transcription,
            'success': True
        })
    except Exception as e:
        logging.error(f"Transcription failed: {str(e)}")
        return jsonify({
            'error': f'Transcription failed: {str(e)}',
            'success': False
        }), 500

@app.route('/process', methods=['POST'])
def process_text():
    
    if not request.is_json:
        logging.error("Content-Type must be application/json")
        return jsonify({'error': 'Content-Type must be application/json'}), 415
        
    try:
        data = request.get_json()
        if data is None:
            logging.error("Invalid JSON format")
            return jsonify({'error': 'Invalid JSON format'}), 400
            
        text = data.get('text', '')
        is_audio = data.get('is_audio', False)
        chat_id = data.get('chat_id')

        if not text.strip():
            logging.error("No text provided for summarization")
            return jsonify({'error': 'No text provided for summarization'}), 400

        if not chat_id or chat_id == 'null':
            chat_id = str(uuid.uuid4())
            logging.debug(f"Generated new chat_id: {chat_id}")

        user_id = request.remote_addr
        logging.debug(f"Processing text for user_id: {user_id}, chat_id: {chat_id}, is_audio: {is_audio}")
        result = generate_abstractive_content(text, summary_ratio=0.4, is_audio=is_audio)
        logging.debug(f"generate_abstractive_content result: {result}")

        if 'error' in result['abstractive_report'].lower():
            logging.error(f"Abstractive summary failed: {result['abstractive_report']}")
            return jsonify({'error': result['abstractive_report']}), 500

        from chatbot import DATABASE
        try:
            with sqlite3.connect(DATABASE) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT id FROM Chat WHERE id = ?', (chat_id,))
                if not cursor.fetchone():
                    cursor.execute('INSERT INTO Chat (id, user_id) VALUES (?, ?)', (chat_id, user_id))
                    logging.debug(f"Inserted new chat: {chat_id}")
                cursor.execute('INSERT INTO Message (chat_id, role, content) VALUES (?, ?, ?)', 
                             (chat_id, 'user', text))
                cursor.execute('INSERT INTO Message (chat_id, role, content) VALUES (?, ?, ?)', 
                             (chat_id, 'assistant', result['abstractive_report']))
                conn.commit()
                logging.info(f"Messages saved to database for chat_id: {chat_id}")
        except sqlite3.Error as db_error:
            logging.error(f"Database error: {str(db_error)}")
            return jsonify({'error': f"Database error: {str(db_error)}"}), 500

        summaries[user_id] = {
            'input_text': text,
            'extractive_summary': result['extractive_summary'],
            'abstractive_report': result['abstractive_report']
        }

        logging.info(f"Text processed successfully for chat_id: {chat_id}")
        return jsonify({
            'extractive_summary': result['extractive_summary'],
            'abstractive_report': result['abstractive_report'],
            'chat_id': chat_id
        })
    except Exception as e:
        logging.error(f"Error in /process: {str(e)}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/chat', methods=['POST'])
def chat():
    
    if not request.is_json:
        logging.error("Content-Type must be application/json")
        return jsonify({'error': 'Content-Type must be application/json'}), 415
        
    try:
        data = request.get_json()
        if data is None:
            logging.error("Invalid JSON format")
            return jsonify({'error': 'Invalid JSON format'}), 400
            
        user_id = request.remote_addr
        user_message = data.get('message', '').strip()
        chat_id = data.get('chat_id', str(uuid.uuid4()))

        if not user_message:
            logging.error("Empty message")
            return jsonify({'error': 'Empty message'}), 400

        response, chat_id = chat_with_model(user_id, user_message, chat_id, summaries)
        if response.startswith('Error'):
            logging.error(f"Chat failed: {response}")
            return jsonify({'error': response}), 500

        return jsonify({'response': response, 'chat_id': chat_id})
    except Exception as e:
        logging.error(f"Error in /chat: {str(e)}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/chat-history', methods=['GET'])
def get_chat_history_route():
    
    try:
        user_id = request.remote_addr
        chat_history = get_chat_history(user_id)
        if any('error' in chat for chat in chat_history):
            logging.error(f"Chat history retrieval failed: {chat_history[0]['error']}")
            return jsonify({'error': chat_history[0]['error']}), 500

        return jsonify(chat_history)
    except Exception as e:
        logging.error(f"Error retrieving chat history: {str(e)}")
        return jsonify({'error': f'Error retrieving chat history: {str(e)}'}), 500

@app.route('/load-chat', methods=['POST'])
def load_chat_route():
    
    try:
        data = request.get_json()
        chat_id = data.get('chat_id')

        result = load_chat(chat_id)
        if 'error' in result:
            logging.error(f"Chat load failed: {result['error']}")
            return jsonify({'error': result['error']}), 404 if result['error'] == 'Chat not found' else 500

        return jsonify(result)
    except Exception as e:
        logging.error(f"Error loading chat: {str(e)}")
        return jsonify({'error': f'Error loading chat: {str(e)}'}), 500

@app.route('/extract-text', methods=['POST'])
def extract_text_route():
    
    if 'file' not in request.files:
        logging.error("No file provided")
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        logging.error("No file selected")
        return jsonify({'error': 'No file selected'}), 400

    text = extract_text(file, file.filename)
    if text.startswith('Error') or text == 'Unsupported file format':
        logging.error(f"Text extraction failed: {text}")
        return jsonify({'error': text}), 400 if text == 'Unsupported file format' else 500

    return jsonify({'text': text})

@app.route('/delete-chat', methods=['POST'])
def delete_chat():
    data = request.get_json()
    chat_id = data.get('chat_id')

    if not chat_id:
        logging.error("Chat ID is required")
        return jsonify({'error': 'Chat ID is required'}), 400

    try:
        with sqlite3.connect('chat_history.db') as conn:
            cursor = conn.cursor()
            # Delete messages associated with the chat
            cursor.execute('DELETE FROM Message WHERE chat_id = ?', (chat_id,))
            # Delete the chat itself
            cursor.execute('DELETE FROM Chat WHERE id = ?', (chat_id,))
            conn.commit()
            logging.info(f"Chat deleted successfully, chat_id: {chat_id}")
        return jsonify({'message': 'Chat deleted successfully'}), 200
    except Exception as e:
        logging.error(f"Error deleting chat: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)