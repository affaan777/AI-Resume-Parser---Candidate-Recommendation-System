"""
Configuration for embeddings module
"""
import os
from dotenv import load_dotenv

load_dotenv()

# # OpenRouter API Configuration (for embeddings)
# OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Embedding Model Configuration
# EMBEDDING_MODEL = "openai/text-embedding-3-small"  # Via OpenRouter
# EMBEDDING_DIMENSIONS = 1536  # Dimensions for text-embedding-3-small

EMBEDDING_MODEL = "models/text-embedding-004"
EMBEDDING_DIMENSIONS = 768  # Gemini uses 768, not 1536

#Pinecone Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "resumes")

# Cache Configuration (Optional - for later)
REDIS_ENABLED = False  # We'll enable this later
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
CACHE_TTL = 2592000  # 30 days in seconds

# Preprocessing Configuration
MAX_TEXT_LENGTH = 8000  # Max characters for embedding
FIELDS_TO_INCLUDE = [
    "professional_summary",
    "skills",
    "work_experience",
    "education",
    "certifications",
    "projects"
]


#Validation
def validate_config():
    """Validate that required configuration is present"""
    errors=[]

    if not GOOGLE_API_KEY:
        errors.append("GOOGLE_API_KEY not found in .env")
    
    if not PINECONE_API_KEY:
        errors.append("PINECONE_API_KEY not found in .env")
    
    if errors:
        raise ValueError(f"Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))
    
    return True

# Validate on import
validate_config()
print("✅ Configuration loaded successfully")
