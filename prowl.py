from flask import Flask, request, jsonify, render_template
from flask_basicauth import BasicAuth
import chromadb
import html
import requests
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import os
from openai import OpenAI
from dotenv import load_dotenv
import logging
import traceback
from typing import List, Dict, Optional
from functools import lru_cache
from datetime import datetime, timedelta
from time import sleep

# ------------------------------------------------------ LOGGING/INIT ---------------------------------------------------------------- #

# Logging setup
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

# Database configuration
DB_URL = "https://storage.googleapis.com/prowl_database/chroma_db/database.chroma"
DB_DIR = "chroma_db"
DB_PATH = os.path.join(DB_DIR, "database.chroma")

def download_database():
    if not os.path.exists(DB_PATH):
        os.makedirs(DB_DIR, exist_ok=True)
        print("Downloading database from cloud storage...")
        
        response = requests.get(DB_URL, stream=True)
        with open(DB_PATH, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        print("Database downloaded successfully!")

download_database()

class RateLimiter:
    def __init__(self, max_tokens_per_min=150000):
        self.max_tokens_per_min = max_tokens_per_min
        self.tokens_used = 0
        self.last_reset = datetime.now()

    def check_and_wait(self, estimated_tokens):
        now = datetime.now()
        if now - self.last_reset > timedelta(minutes=1):
            self.tokens_used = 0
            self.last_reset = now
            
        if self.tokens_used + estimated_tokens > self.max_tokens_per_min:
            sleep_time = 60 - (now - self.last_reset).seconds
            sleep(sleep_time)
            self.tokens_used = 0
            self.last_reset = datetime.now()
            
        self.tokens_used += estimated_tokens

rate_limiter = RateLimiter()

try:
    logger.info("Initializing ChromaDB...")
    
    chroma_client = chromadb.Client(Settings(
        persist_directory=os.path.abspath(DB_DIR),
        is_persistent=True
    ))
    
    @lru_cache(maxsize=1000)
    def get_cached_embedding(text: str) -> List[float]:
        estimated_tokens = len(text) // 4
        rate_limiter.check_and_wait(estimated_tokens)
        return openai_ef([text])[0]

    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key,
        model_name="text-embedding-ada-002"
    )

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
    
    
# ------------------------------------------------------ OPENAI INSTRUCTIONS ---------------------------------------------------------------- #
    

def get_relevant_context(query: str, n_results: int = 3) -> List[Dict]:

    try:
        logger.debug(f"Querying ChromaDB with: {query}")
        query_embedding = get_cached_embedding(query)
        
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )
        
        if not results['documents'][0]:
            logger.warning("No results found in ChromaDB")
            return []
            
        combined_results = {}
        max_chunk_length = 1000
        
        for idx, doc in enumerate(results['documents'][0]):
            metadata = results['metadatas'][0][idx]
            parent_id = metadata['parent_record']
            
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
        log_token_usage(query, result_list)
        return result_list
    
    except Exception as e:
        logger.error(f"Error retrieving context: {str(e)}\n{traceback.format_exc()}")
        return []

class ConversationState:
    def __init__(self):
        self.current_collection: Optional[Dict] = None
        self.previous_collection: Optional[Dict] = None
        self.last_query: Optional[str] = None
        self.last_response: Optional[str] = None

    def update_collection(self, new_collection: Optional[Dict]):
        
        logger.debug(f"Updating collection - Current before: {self.current_collection is not None}")
        logger.debug(f"Updating collection - Previous before: {self.previous_collection is not None}")
        
        if new_collection != self.current_collection:
            self.previous_collection = self.current_collection
            self.current_collection = new_collection
            
        logger.debug(f"Updating collection - Current after: {self.current_collection is not None}")
        logger.debug(f"Updating collection - Previous after: {self.previous_collection is not None}")  

    def get_active_collection(self) -> Optional[Dict]:
        
        logger.debug(f"Getting active collection - Current: {self.current_collection is not None}")
        logger.debug(f"Getting active collection - Previous: {self.previous_collection is not None}")
        
        return self.current_collection or self.previous_collection

conversation_state = ConversationState()

def filter_query(query: str) -> bool:
    prohibited_keywords = ['hack', 'illegal', 'attack', 'exploit']
    return not any(keyword in query.lower() for keyword in prohibited_keywords)

def sanitize_input(query: str) -> str:
    return html.escape(query.strip())

def get_greeting_response(query: str) -> str:
    greetings = ['hello', 'hi', 'hi!', 'hey!', 'hey', 'sup', 'good morning', 'good afternoon', 'good evening', 'yo', 'yo!']
    if query.lower().strip() in greetings:
        return "Hi! How may I help you today?"
    return None

def get_faq_response(query: str) -> str:
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

def get_negative_response(query: str) -> str:
    negatives = ['no', 'nah', 'nope', 'negative', 'i dont think so']
    if query.lower().strip() in negatives:
        return "Okay. Is there anything else I can help you with today?"
    return None

def is_followup_query(query: str) -> bool:
    followup_phrases = {
        'yes', 'yes!', 'yes please', 'sure', 'i would', 'yes please!', 'yup', 'yup!',
        'yes i would', 'yes i would!', 'tell me more',
        'can you tell me more', 'id like more information'
    }
    return query.lower().strip() in followup_phrases and conversation_state.get_active_collection() is not None

def format_collection_context(collection: Dict) -> str:
    return f"""
Title: {collection['metadata']['Title/Dates']}
Content: {collection['content']}
Call Number: {collection['metadata'].get('Call Number', 'Not available')}
{"=" * 50}
"""

def generate_response(query: str, context: List[Dict]) -> str:
    try:
        logger.debug(f"Generate response - Query: {query}")
        logger.debug(f"Generate response - Context length: {len(context)}")
        logger.debug(f"Generate response - Current collection before: {conversation_state.current_collection is not None}")
        
        conversation_state.last_query = query
        
        negative_response = get_negative_response(query)
        if negative_response:
            return negative_response

        greeting_response = get_greeting_response(query)
        if greeting_response:
            return greeting_response

        faq_response = get_faq_response(query)
        if faq_response:
            return faq_response
            
        if context:
           logger.debug("Updating conversation state with new context")
           conversation_state.update_collection({
               'metadata': context[0]['metadata'],
               'content': context[0]['content']
            })
            
        logger.debug(f"Generate response - Current collection after update: {conversation_state.current_collection is not None}")

        if is_followup_query(query):
            active_collection = conversation_state.get_active_collection()
            if active_collection:
                if conversation_state.last_response and "Would you like to schedule a visit" in conversation_state.last_response:
                    return "Great! You can schedule your visit from https://uwm.edu/libraries/forms/visit-distinctive-collections/ or call us at (414) 229-5402. Our regular hours are Monday through Friday, 9:00 AM to 4:30 PM."
                
                logger.debug("Formatting context for follow-up")
                formatted_context = format_collection_context(active_collection)
                prompt = f"""You are a helpful university archives assistant. The user wants more information about a previously mentioned collection. Please:

1. Focus on different aspects than what was previously mentioned
2. Include specific details about the collection
3. Try not to use words like "fascinating"
4. End by asking if they would like to schedule a visit to the archives
5. Keep the response under 3 sentences
6. Don't mention details that were already shared
7. Don't use the term "Finding Aid"

Previous Response: {conversation_state.last_response}

Context:
{formatted_context}

Generate a new, non-repetitive response about this specific collection:"""
            else:
                logger.debug("No active collection found for follow-up")
                return "I'm not sure which collection you're asking about. Could you please specify?"
        else:
            formatted_context = ""
            if context:
                conversation_state.update_collection({
                    'metadata': context[0]['metadata'],
                    'content': context[0]['content']
                })
                
                for item in context:
                    formatted_context += format_collection_context(item)

            prompt = f"""You are a helpful university archives assistant. Please:

1. Start responses with "I have" or "We have" when referring to collections
2. Provide just the basic title, date range, and a one-phrase description
3. Ask if the user would like more information about the collection
4. Avoid using the term "Finding Aid"
5. Keep responses under 3 sentences

Context:
{formatted_context}

User Question: {query}

Generate a concise response:"""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an archives assistant who gives brief, non-repetitive responses."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=150
        )
        
        final_response = response.choices[0].message.content
        conversation_state.last_response = final_response
        return final_response

    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
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
            
        logger.debug(f"Ask route - Message: {user_message}")
        logger.debug(f"Ask route - Current collection before: {conversation_state.current_collection is not None}")
        
        context = get_relevant_context(user_message)
        response = generate_response(user_message, context)
        
        response = generate_response(user_message, context)
        
        logger.debug(f"Ask route - Current collection after: {conversation_state.current_collection is not None}")
        
        logger.info(f"User Message: {user_message}")
        logger.info(f"Bot Response: {response}")
        
        return jsonify({'response': response})
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/')
def home():
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering template: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': 'Template error'}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)