# RAG Configuration File
# Update these settings as needed

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# LLM Provider Settings
USE_GEMINI = False  # Set to True to use Gemini, False to use OpenAI GPT
USE_GPT = True      # Set to True to use OpenAI GPT, False to use Gemini

# OpenAI Settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")  # Set your OpenAI API key here or use environment variable
OPENAI_MODEL = "gpt-3.5-turbo"

# Gemini Settings
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")  # Set your Gemini API key here or use environment variable
GEMINI_MODEL = "gemini-1.5-flash"

GROQ_API_KEY = os.getenv("GROQ_API_KEY") 
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-70b-8192")

GROQ_API_KEY_LITE = os.getenv("GROQ_API_KEY_LITE") 
GROQ_MODEL_LITE = "llama3-8b-8192"

# Common LLM Settings
MAX_TOKENS = 1200  # Maximum tokens for LLM response generation (increased for more detailed answers)
TEMPERATURE = 0.4

# API Authentication
BEARER_TOKEN = "c6cee5b5046310e401632a7effe9c684d071a9ef5ce09b96c9ec5c3ebd13085e"

# Retrieval Settings
TOP_K = 12  # 3 queries * 3 results per query = 9 total candidates
SCORE_THRESHOLD = 0.3  # Lowered for better recall
RERANK_TOP_K =  7  # Final number after reranking
BM25_WEIGHT = 0.3  # Weight for BM25 in hybrid search
SEMANTIC_WEIGHT = 0.7  # Weight for semantic search

# Advanced RAG Settings
ENABLE_RERANKING = True
ENABLE_HYBRID_SEARCH = True
ENABLE_QUERY_EXPANSION = True
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
QUERY_EXPANSION_COUNT = 3  # Number of focused sub-queries to generate from complex questions
SCORE_THRESHOLD = 0.3  # Threshold for semantic search (lowered for better recall)
MAX_CONTEXT_LENGTH = 4000

# Query Expansion Retrieval Strategy
USE_TOTAL_BUDGET_APPROACH = True  # If True, distribute TOP_K across all queries; if False, use TOP_K per query

# Model Settings
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 4

# Database Settings
OUTPUT_DIR = os.getenv("RAG_EMBEDDINGS_PATH", "./RAG/rag_embeddings")

# Advanced Settings
CHUNK_SIZE = 400
CHUNK_OVERLAP = 100

# API Settings
API_HOST = "0.0.0.0"
API_PORT = 8000
API_RELOAD = True

assert OPENAI_API_KEY, "OPENAI KEY NOT SET"
assert GEMINI_API_KEY, "GEMINI KEY NOT SET"
assert GROQ_API_KEY, "GROQ KEY NOT SET"
assert GROQ_API_KEY_LITE, "GROQ KEY LITE NOT SET"

def get_provider_configs():
    """
    Get configurations for all provider instances.
    You can configure multiple instances of each provider type.
    """
    configs = {
        "groq": [],
        "gemini": [],
        "openai": []
    }
    
    # Groq configurations
    # You can add multiple Groq instances with different API keys
    groq_configs = [
        {
            "name": "primary",
            "api_key": os.getenv("GROQ_API_KEY_1"),
            "model": os.getenv("GROQ_MODEL_1", "qwen/qwen3-32b")
        },
        {
            "name": "secondary", 
            "api_key": os.getenv("GROQ_API_KEY_2"),
            "model": os.getenv("GROQ_MODEL_2", "qwen/qwen3-32b")
        }
    ]
    
    # Only add Groq configs that have API keys
    for config in groq_configs:
        if config["api_key"]:
            configs["groq"].append(config)
    
    # Gemini configurations
    # You can add multiple Gemini instances with different API keys
    gemini_configs = [
        {
            "name": "primary",
            "api_key": os.getenv("GEMINI_API_KEY_1"),
            "model": os.getenv("GEMINI_MODEL_1", "gemini-1.5-flash")
        },
        {
            "name": "secondary",
            "api_key": os.getenv("GEMINI_API_KEY_2"), 
            "model": os.getenv("GEMINI_MODEL_2",  "gemini-1.5-flash")
        },
        {
            "name": "tertiary",
            "api_key": os.getenv("GEMINI_API_KEY_3"),
            "model": os.getenv("GEMINI_MODEL_3",  "gemini-1.5-flash")
        }
    ]
    
    # Only add Gemini configs that have API keys
    for config in gemini_configs:
        if config["api_key"]:
            configs["gemini"].append(config)
    
    # OpenAI configurations
    # You can add multiple OpenAI instances with different API keys
    openai_configs = [
    ]
    
    # Only add OpenAI configs that have API keys
    for config in openai_configs:
        if config["api_key"]:
            configs["openai"].append(config)
    
    return configs