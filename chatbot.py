import sqlite3
import uuid
import ollama
import logging
import re

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Database file path
DATABASE = 'chat_history.db'

# Global chat cache
chat_cache = {}

# Initialize database
def init_db():
    
    try:
        with sqlite3.connect(DATABASE) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS Chat (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS Message (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    FOREIGN KEY (chat_id) REFERENCES Chat (id)
                )
            ''')
            conn.commit()
            logging.info("Database initialized successfully")
    except sqlite3.Error as e:
        logging.error(f"Database initialization failed: {str(e)}")
        raise

init_db()

def chat_with_model(user_id, user_message, chat_id, summaries):
    try:
        # Ensure chat_id is valid
        if not chat_id or chat_id == 'null':
            chat_id = str(uuid.uuid4())
            logging.debug(f"Generated new chat_id: {chat_id}")

        with sqlite3.connect(DATABASE) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT id FROM Chat WHERE id = ?', (chat_id,))
            chat = cursor.fetchone()
            if not chat:
                cursor.execute('INSERT INTO Chat (id, user_id) VALUES (?, ?)', (chat_id, user_id))
                conn.commit()
                logging.debug(f"Inserted new chat: {chat_id}")

            # Save user message
            cursor.execute('INSERT INTO Message (chat_id, role, content) VALUES (?, ?, ?)', 
                         (chat_id, 'user', user_message))
            conn.commit()
            logging.debug(f"User message saved: {user_message}")

            # Check cache
            cache_key = f"{user_id}:{chat_id}:{user_message}"
            if cache_key in chat_cache:
                response = chat_cache[cache_key]
                logging.debug("Using cached response")
            else:
                # Get previous summary
                prev_summary = summaries.get(user_id, {
                    'extractive_summary': '',
                    'abstractive_report': '',
                    'input_text': ''
                })
                extractive = prev_summary['extractive_summary']
                abstractive = prev_summary['abstractive_report']
                input_text = prev_summary['input_text']

                # Get recent chat history (last 3 messages for context)
                cursor.execute('SELECT role, content FROM Message WHERE chat_id = ? ORDER BY id DESC LIMIT 3', 
                             (chat_id,))
                recent_messages = [{'role': row[0], 'content': row[1]} for row in cursor.fetchall()]
                chat_context = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" 
                                       for msg in recent_messages[::-1]])

                # Determine user intent
                user_message_lower = user_message.lower()
                intent = "general"
                if any(keyword in user_message_lower for keyword in ["shorter", "simplify", "concise"]):
                    intent = "shorten_summary"
                elif any(keyword in user_message_lower for keyword in ["more details", "expand", "elaborate"]):
                    intent = "expand_summary"
                elif any(keyword in user_message_lower for keyword in ["what is", "tell me about", "describe"]):
                    intent = "informational"
                elif any(keyword in user_message_lower for keyword in ["why", "how", "explain"]):
                    intent = "explanation"

                # Build prompt
                prompt = f"""
### System Instructions ###
You are a helpful assistant for a text summarization tool, designed to provide accurate, concise, and contextually relevant responses. Follow these guidelines:
- Adopt a professional yet approachable tone.
- Keep responses concise: 2-4 sentences for questions, 100-200 words for summaries.
- Use the provided context (input text, summaries, chat history) only when relevant to the user's query.
- Ensure factual accuracy and avoid speculation.
- If the query is unclear or off-topic, respond with: "I'm not sure how to help with that. Could you clarify?"
- Structure responses clearly, using bullet points or paragraphs as appropriate.
- Remove any <think> tags from the output.

### Context ###
- **Original Input Text**: {input_text[:1000] + '...' if len(input_text) > 1000 else input_text}
- **Extractive Summary**: {extractive[:500] + '...' if len(extractive) > 500 else extractive}
- **Abstractive Summary**: {abstractive[:500] + '...' if len(abstractive) > 500 else abstractive}
- **Recent Chat History**:
{chat_context if chat_context else "No recent messages."}

### User Query ###
{user_message}

### Task ###
Based on the user's query and context, perform one of the following:

1. **Shorten/Simplify Summary** (if requested, e.g., "make it shorter", "simplify"):
   - Revise the abstractive summary to be 50-100 words, preserving key points.
   - Use simple language and avoid jargon.
   - Example: "Simplify this summary" → Shorten to: "The article discusses AI advancements, focusing on efficiency."

2. **Expand Summary** (if requested, e.g., "more details", "expand"):
   - Elaborate on the abstractive summary, adding 100-150 words with insights from the input text and extractive summary.
   - Include relevant details and context.
   - Example: "Expand the summary" → Add: "The article details AI models like GPT, highlighting training efficiency..."

3. **Informational Query** (e.g., "What is X?", "Tell me about Y"):
   - Provide a concise answer (2-4 sentences) using the context if relevant, or general knowledge if not.
   - Example: "What is AI?" → "AI is technology enabling machines to perform tasks like reasoning, using algorithms and data."

4. **Explanation Query** (e.g., "Why X?", "How does Y work?"):
   - Offer a clear explanation (3-5 sentences) grounded in the context or general knowledge.
   - Example: "How does summarization work?" → "Summarization uses NLP to extract or generate key points from text..."

5. **General Query**:
   - Interpret the query naturally and respond based on the context and chat history.
   - Example: "Can you help with this?" → "Please provide more details about what you need help with."

### Response ###
Provide the response below, adhering to the guidelines and task instructions.
"""
                try:
                    response = ollama.generate(
                        model='deepseek-r1:1.5b',
                        prompt=prompt,
                        stream=False
                    )
                    chat_response = response.get('response', 'Error: No response from model.')
                    chat_response = re.sub(r'<think>.*?</think>', '', chat_response, flags=re.DOTALL).strip()
                    if not chat_response:
                        chat_response = "I'm not sure how to help with that. Could you clarify?"
                    chat_cache[cache_key] = chat_response
                    logging.info(f"Chat response generated: {chat_response[:100]}...")
                except Exception as e:
                    logging.error(f"Error generating chat response: {str(e)}")
                    chat_response = "Error: Unable to generate response. Please try again."

                response = chat_response

            # Save assistant response
            cursor.execute('INSERT INTO Message (chat_id, role, content) VALUES (?, ?, ?)', 
                         (chat_id, 'assistant', response))
            conn.commit()
            logging.info(f"Assistant response saved to database for chat_id: {chat_id}")

        return response, chat_id
    except Exception as e:
        logging.error(f"Error in chat generation: {str(e)}")
        return f"Error: {str(e)}", chat_id

def get_chat_history(user_id):
    
    try:
        with sqlite3.connect(DATABASE) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT id FROM Chat WHERE user_id = ?', (user_id,))
            chats = cursor.fetchall()

            chat_history = []
            for chat in chats:
                chat_id = chat[0]
                cursor.execute('SELECT role, content FROM Message WHERE chat_id = ?', (chat_id,))
                messages = [{'role': row[0], 'content': row[1]} for row in cursor.fetchall()]
                chat_history.append({'chat_id': chat_id, 'messages': messages})

        logging.info(f"Chat history retrieved for user_id: {user_id}")
        return chat_history
    except Exception as e:
        logging.error(f"Error retrieving chat history: {str(e)}")
        return [{'error': f'Error retrieving chat history: {str(e)}'}]

def load_chat(chat_id):
    
    try:
        with sqlite3.connect(DATABASE) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT role, content FROM Message WHERE chat_id = ?', (chat_id,))
            messages = [{'role': row[0], 'content': row[1]} for row in cursor.fetchall()]
        
        if not messages:
            logging.error(f"Chat not found: {chat_id}")
            return {'error': 'Chat not found'}

        logging.info(f"Chat loaded: {chat_id}")
        return {'chat_id': chat_id, 'messages': messages}
    except Exception as e:
        logging.error(f"Error loading chat: {str(e)}")
        return {'error': f'Error loading chat: {str(e)}'}
    
