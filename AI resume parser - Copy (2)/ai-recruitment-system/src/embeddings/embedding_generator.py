"""
Embedding Generator: Convert text to embeddings using Google Gemini
"""
from typing import List, Optional
import time
import google.generativeai as genai
from src.embeddings.config import (
    GOOGLE_API_KEY,
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSIONS
)


class EmbeddingGenerator:
    """Generate embeddings using Google Gemini API"""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize embedding generator
        
        Args:
            api_key: Google API key (uses config default if not provided)
            model: Embedding model to use (uses config default if not provided)
        """
        self.api_key = api_key or GOOGLE_API_KEY
        self.model = model or EMBEDDING_MODEL
        self.dimensions = EMBEDDING_DIMENSIONS
        
        if not self.api_key:
            raise ValueError("Google API key is required")
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
    
    def generate_embedding(self, text: str, task_type: str = "retrieval_document") -> List[float]:
        """
        Generate embedding for a single text
        
        Args:
            text: Input text to embed
            task_type: Task type for embedding
                - "retrieval_document": For documents in a retrieval system
                - "retrieval_query": For search queries
                - "semantic_similarity": For semantic similarity tasks
                - "classification": For classification tasks
            
        Returns:
            List of floats representing the embedding vector
        """
        try:
            result = genai.embed_content(
                model=self.model,
                content=text,
                task_type=task_type
            )
            
            embedding = result['embedding']
            
            # Validate embedding dimensions
            if len(embedding) != self.dimensions:
                raise ValueError(
                    f"Expected {self.dimensions} dimensions, got {len(embedding)}"
                )
            
            return embedding
            
        except Exception as e:
            raise Exception(f"Embedding generation failed: {str(e)}")
    
    def generate_embeddings_batch(
        self, 
        texts: List[str],
        task_type: str = "retrieval_document",
        delay_seconds: float = 0.1
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts
        
        Args:
            texts: List of input texts
            task_type: Task type for embeddings
            delay_seconds: Delay between requests to avoid rate limits
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        
        for i, text in enumerate(texts):
            try:
                embedding = self.generate_embedding(text, task_type=task_type)
                embeddings.append(embedding)
                
                # Progress indicator
                if (i + 1) % 10 == 0:
                    print(f"Generated {i + 1}/{len(texts)} embeddings")
                
                # Rate limiting
                if delay_seconds > 0 and i < len(texts) - 1:
                    time.sleep(delay_seconds)
                    
            except Exception as e:
                print(f"Error generating embedding for text {i + 1}: {e}")
                # Return None for failed embeddings
                embeddings.append(None)
        
        return embeddings
    
    def get_embedding_info(self) -> dict:
        """Get information about the embedding configuration"""
        return {
            "model": self.model,
            "dimensions": self.dimensions,
            "provider": "Google Gemini"
        }


# Utility function for quick embedding generation
def generate_embedding(text: str) -> List[float]:
    """
    Quick utility to generate a single embedding
    
    Args:
        text: Text to embed
        
    Returns:
        Embedding vector
    """
    generator = EmbeddingGenerator()
    return generator.generate_embedding(text)


if __name__ == "__main__":
    # Test the embedding generator
    print("🧪 Testing Embedding Generator (Gemini)...")
    print("=" * 80)
    
    try:
        generator = EmbeddingGenerator()
        
        # Print config info
        info = generator.get_embedding_info()
        print(f"\n✅ Configuration:")
        print(f"   Provider: {info['provider']}")
        print(f"   Model: {info['model']}")
        print(f"   Dimensions: {info['dimensions']}")
        
        # Test with sample text
        test_text = "Experienced Python developer with expertise in machine learning and FastAPI"
        
        print(f"\n🔄 Generating embedding for test text...")
        print(f"   Text: {test_text[:80]}...")
        
        start_time = time.time()
        embedding = generator.generate_embedding(test_text)
        elapsed = time.time() - start_time
        
        print(f"\n✅ Embedding generated successfully!")
        print(f"   Dimensions: {len(embedding)}")
        print(f"   Time taken: {elapsed:.3f}s")
        print(f"   First 10 values: {[f'{x:.4f}' for x in embedding[:10]]}")
        print(f"   Data type: {type(embedding[0])}")
        
        # Test batch generation
        print(f"\n🔄 Testing batch generation (3 texts)...")
        test_texts = [
            "Python developer with ML experience",
            "Java backend engineer with Spring Boot",
            "Frontend developer skilled in React and TypeScript"
        ]
        
        start_time = time.time()
        embeddings = generator.generate_embeddings_batch(test_texts, delay_seconds=0.1)
        elapsed = time.time() - start_time
        
        successful = sum(1 for e in embeddings if e is not None)
        print(f"\n✅ Batch generation complete!")
        print(f"   Successful: {successful}/{len(test_texts)}")
        print(f"   Total time: {elapsed:.3f}s")
        print(f"   Avg time per embedding: {elapsed/len(test_texts):.3f}s")
        
        # Test query vs document embeddings
        print(f"\n🔄 Testing query embedding...")
        query_text = "Looking for Python ML engineer"
        query_embedding = generator.generate_embedding(query_text, task_type="retrieval_query")
        print(f"   Query embedding generated: {len(query_embedding)} dimensions")
        
        print("\n" + "=" * 80)
        print("✅ All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()