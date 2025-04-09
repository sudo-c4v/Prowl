from flask import Flask, request, jsonify, render_template, session
from flask_basicauth import BasicAuth
import chromadb
import html
import requests
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import os
import tiktoken
from openai import OpenAI
from dotenv import load_dotenv
import logging
import traceback
from typing import List, Dict, Optional
from functools import lru_cache
from datetime import datetime, timedelta
from time import time, sleep
from sklearn.metrics.pairwise import cosine_similarity

# ------------------------------------------------------ LOGGING/INIT ---------------------------------------------------------------- #

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chatbot_debug.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logger.error("OpenAI API key not found in environment variables")
    raise ValueError("OpenAI API key not found!")

app = Flask(__name__, static_folder='static', static_url_path='/static')
app.secret_key = os.urandom(24)
client = OpenAI(api_key=api_key)

# ------------------------------------------------------ DATABASE SETUP ---------------------------------------------------------------- #

# DB Connection
CHROMADB_HOST = os.getenv("CHROMADB_HOST", "prowlreg.azurecr.io/chromadb-server.io")
CHROMADB_PORT = int(os.getenv("CHROMADB_PORT", "8000"))

# Initialize the HTTP client
logger.info(f"Connecting to ChromaDB at {CHROMADB_HOST}:{CHROMADB_PORT}")
client = chromadb.HttpClient(
    host=CHROMADB_HOST,
    port=CHROMADB_PORT
)

# ------------------------------------------------------ RATE LIMITER ---------------------------------------------------------------- #

class RateLimiter:
    def __init__(self, max_tokens_per_min=150000):
        self.max_tokens_per_min = max_tokens_per_min
        self.tokens_used = 0
        self.last_reset = time()
        self.tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")


    def count_tokens(self, text: str) -> int:
        """Accurately count tokens using OpenAI's tokenizer."""
        return len(self.tokenizer.encode(text))

    def check_and_wait(self, estimated_tokens):
        """Enforce the token limit per minute."""
        now = time()
        elapsed = now - self.last_reset

        if elapsed > 60:  # Reset every minute
            self.tokens_used = 0
            self.last_reset = now

        if self.tokens_used + estimated_tokens > self.max_tokens_per_min:
            sleep_time = 60 - elapsed
            logger.warning(f"Rate limit hit! Sleeping for {sleep_time:.2f} seconds.")
            sleep(sleep_time)
            self.tokens_used = 0
            self.last_reset = time()

        self.tokens_used += estimated_tokens

rate_limiter = RateLimiter()

# ------------------------------------------------------ CHROMADB AND EMBEDDING FUNCTIONS ---------------------------------------------------------------- #

try:
    logger.info("Initializing ChromaDB...")
    chroma_client = chromadb.Client(Settings(
        persist_directory=os.path.abspath(DB_DIR),
        is_persistent=True
    ))
    
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name="text-embedding-ada-002"
    )

    @lru_cache(maxsize=1000)
    def get_cached_embedding(text: str) -> List[float]:
        estimated_tokens = rate_limiter.count_tokens(text)
        
        rate_limiter.check_and_wait(estimated_tokens)
        return openai_ef([text])[0]

    collection = chroma_client.get_or_create_collection(
        name="archive_data",
        embedding_function=openai_ef
    )
    
    collection_count = len(collection.get()["ids"]) if collection.get()["ids"] else 0
    logger.info(f"Connected to ChromaDB collection with {collection_count} records")

except Exception as e:
    logger.error(f"Error initializing ChromaDB: {str(e)}\n{traceback.format_exc()}")
    raise

def log_token_usage(query: str, context: List[Dict]):
    query_tokens = len(query) // 4
    context_tokens = sum(len(str(c)) for c in context) // 4
    logger.info(f"Estimated tokens - Query: {query_tokens}, Context: {context_tokens}")
    
    
def truncate_context(context: str, max_tokens: int = 500) -> str:
    tokens = context.split()
    return " ".join(tokens[:max_tokens]) + " [...]" if len(tokens) > max_tokens else context

    
def get_relevant_context(query: str, n_results: int = 3) -> List[Dict]:
    try:
        normalized_query = query.strip().lower()
        logger.debug(f"Querying ChromaDB with: {normalized_query}")
        
        # Get query embedding for similarity search
        query_embedding = get_cached_embedding(normalized_query)
        
        logger.debug(f"Query embedding length: {len(query_embedding)}")
        
        # Run semantic search using ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        # Log the raw results from ChromaDB.
        logger.debug(f"Raw ChromaDB results: {results}")

        if not results['documents'][0]:
            logger.warning("No results found in ChromaDB for the query.")
            return []
        
        combined_results = {}
        max_chunk_length = 1000  # Avoid overloading tokens

        for idx, doc in enumerate(results['documents'][0]):
            metadata = results['metadatas'][0][idx]
            parent_id = metadata.get('parent_record', f"doc_{idx}")

            if parent_id not in combined_results:
                combined_results[parent_id] = {
                    'content': doc[:max_chunk_length],
                    'metadata': metadata
                }
            else:
                current_length = len(combined_results[parent_id]['content'])
                if current_length < max_chunk_length:
                    remaining_space = max_chunk_length - current_length
                    combined_results[parent_id]['content'] += f"\n{doc[:remaining_space]}"
        
        result_list = list(combined_results.values())

        log_token_usage(normalized_query, result_list)
        logger.debug(f"Retrieved context: {result_list}")
        return result_list

    except Exception as e:
        logger.error(f"Error retrieving context: {str(e)}\n{traceback.format_exc()}")
        return []


# ------------------------------------------------------ SESSION MANAGEMENT ---------------------------------------------------------------- #

def get_conversation_state() -> dict:
    if 'conversation_state' not in session:
        session['conversation_state'] = {
            'state': 'NEW_QUERY',  # possible states: NEW_QUERY, FOLLOW_UP, CLARIFICATION
            'last_query': None,
            'last_response': None,
            'active_topic': None,  # holds active collection/topic data
            'previous_topic': None
        }
    return session['conversation_state']

def update_conversation_state(new_data: dict):
    state = get_conversation_state()
    state.update(new_data)
    session['conversation_state'] = state

def clear_conversation_state():
    session.pop('conversation_state', None)

# ------------------------------------------------------ INPUT FILTERS AND PREDEFINED RESPONSES ---------------------------------------------------------------- #

def filter_query(query: str) -> bool:
    prohibited_keywords = ['hack', 'illegal', 'attack', 'exploit']
    return not any(keyword in query.lower() for keyword in prohibited_keywords)

def sanitize_input(query: str) -> str:
    return html.escape(query.strip())

def get_greeting_response(query: str) -> Optional[str]:
    greetings = ['hello', 'hi', 'hi!', 'hey!', 'hey', 'sup', 'good morning', 'good afternoon', 'good evening', 'yo', 'yo!']
    if query.lower().strip() in greetings:
        return "Hi! How may I help you today?"
    return None

def get_faq_response(query: str) -> Optional[str]:
    query_lower = query.lower().strip()
    faqs = {
        'hours': {
            'patterns': [
                'what are the hours', 'what are the archives hours', 'hours of operation', 'when are you open', 'are you open',
                'opening hours', 'are the archives open', 'what time', 'what time do you close', 'what time do you open'
            ],
            'response': "The Archives are open Monday through Friday, 9:00 AM to 4:30 PM. We're closed on weekends and federal holidays. Would you like to schedule a visit?"
        },
        'location': {
            'patterns': [
                'where are you located', 'where is', 'location', 'address', 'what is the address',
                'where are the archives', 'how do i get there', 'directions'
            ],
            'response': "We're located on the 3rd floor of the Golda Meir Library at 2311 E Hartford Ave, Milwaukee, WI 53211."           
        },
        'parking': {
            'patterns': ['where can i park', 'parking', 'is there parking'],
            'response': "Visitor parking is available in the Student Union parking structure on N Maryland Ave. Would you like more information about parking options?"
        },
        'appointments': {
            'patterns': ['do i need an appointment', 'can i just show up', 'reservation'],
            'response': "We recommend scheduling your visit in advance to ensure we can best assist you. Would you like the link to our scheduling form?"
        }
    }
    
    for category in faqs.values():
        if any(pattern in query_lower for pattern in category['patterns']):
            return category['response']
    return None

def get_negative_response(query: str) -> Optional[str]:
    negatives = ['no', 'nah', 'nope', 'negative', "i dont think so"]
    if query.lower().strip() in negatives:
        return "Okay. Is there anything else I can help you with today?"
    return None

def get_predefined_response(query: str) -> Optional[str]:
    query_clean = query.lower().strip()
    greetings = {'hello', 'hi', 'hey'}
    thanks = {'thanks', 'thank you'}
    if query_clean in greetings:
        return "Hi! How may I help you today?"
    elif query_clean in thanks:
        return "You're welcome! Is there anything else I can help you with today?"
    return None

def format_collection_context(collection: Dict) -> str:
    """Extracts only the most relevant details from the retrieved collection."""
    metadata = collection['metadata']
    content = collection['content'].split()[:200]  # Limit to 200 words

    return f"""
Title: {metadata.get('Title/Dates', 'Unknown')}
Summary: {" ".join(content)} [...]
Call Number: {metadata.get('Call Number', 'Not available')}
{"=" * 50}
"""

# ------------------------------------------------------ STATE MACHINE FUNCTIONS ---------------------------------------------------------------- #

def query_similarity(query1: str, query2: str) -> float:
    try:
        emb1 = openai_ef([query1])[0]
        emb2 = openai_ef([query2])[0]
        similarity = cosine_similarity([emb1], [emb2])[0][0]
        return similarity
    except Exception as e:
        logger.error(f"Error computing similarity: {str(e)}")
        return 0.0

def determine_query_state(query: str) -> str:
    state = get_conversation_state()
    if state['last_response'] and "Would you like" in state['last_response']:
        return "FOLLOW_UP"
    if state['last_query'] and query_similarity(query, state['last_query']) > 0.75:
        return "CLARIFICATION"
    return "NEW_QUERY"

# ------------------------------------------------------ RESPONSE GENERATION ---------------------------------------------------------------- #

def generate_response(query: str, context: List[Dict]) -> str:
    try:
        state = get_conversation_state()
        new_state = determine_query_state(query)
        logger.debug(f"Determined query state: {new_state}")

        if new_state in ("FOLLOW_UP", "CLARIFICATION") and state.get('active_topic'):
            formatted_context = format_collection_context(state['active_topic'])
        elif context:
            active_topic = {
                'metadata': context[0]['metadata'],
                'content': context[0]['content']
            }
            update_conversation_state({
                'active_topic': active_topic,
                'previous_topic': state.get('active_topic')
            })
            formatted_context = truncate_context("\n".join(format_collection_context(item) for item in context))
        else:
            logger.debug("No context available.")
            return "I'm sorry, I couldn't find any relevant information about that. Could you try rephrasing your question?"
        
        state.update({
            'state': new_state,
            'last_query': query
        })
        session['conversation_state'] = state

        # ✅ Corrected indentation: Now inside the function
        SYSTEM_PROMPT = "You are an archives assistant. Give concise responses under 3 sentences. Prioritize summaries and avoid unnecessary details."

        prompt = f"""
        User Query: {query}

        Context:
        {formatted_context}

        Generate a concise response:
        """

        # ✅ Corrected indentation: Now inside the function
        prompt_tokens = rate_limiter.count_tokens(prompt)
        response_tokens = 150  # Hard cap for response length

        # Enforce rate limit
        rate_limiter.check_and_wait(prompt_tokens + response_tokens)
        
        logger.debug(f"Final prompt to GPT:\n{prompt}")
        
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=150
        )
        
        final_response = response.choices[0].message.content
        state['last_response'] = final_response
        session['conversation_state'] = state
        
        return final_response
        
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}\n{traceback.format_exc()}")
        return "I apologize, but I encountered an error. Could you please try asking your question again?"


# ------------------------------------------------------ ROUTES ---------------------------------------------------------------- #

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.json
        user_message = sanitize_input(data.get('message', ''))

        if not user_message:
            return jsonify({'error': 'No message provided'}), 400

        if not filter_query(user_message):
            return jsonify({'response': "I'm sorry, I can't assist with that."})
        
        predefined = get_predefined_response(user_message)
        if predefined:
            update_conversation_state({'last_query': user_message, 'last_response': predefined})
            return jsonify({'response': predefined})
        
        logger.debug(f"Ask route - Message: {user_message}")
        logger.debug(f"Ask route - Active topic before: {get_conversation_state().get('active_topic') is not None}")
        
        context = get_relevant_context(user_message)
        response_text = generate_response(user_message, context)
        
        logger.debug(f"Ask route - Active topic after: {get_conversation_state().get('active_topic') is not None}")
        logger.info(f"User Message: {user_message}")
        logger.info(f"Bot Response: {response_text}")
        
        return jsonify({'response': response_text})
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500

@app.route('/')
def home():
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering template: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': 'Template error'}), 500

if __name__ == '__main__':
    # Get port from the environment or use 5000 as default
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', debug=False, port=port)
