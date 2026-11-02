"""Gateway-only embeddings. Local/Ollama/backend URL environment values cannot bypass it."""
from model_gateway import encode_batch
EMBEDDING_BACKEND='gateway'

def encode_query(text):
    return encode_batch([text])[0]

def is_remote_backend():return True

def is_ollama_backend():return False
