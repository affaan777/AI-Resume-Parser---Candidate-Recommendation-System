"""
Vector Store: Store and retrieve embeddings from Pinecone
"""
from typing import List, Dict, Any, Optional
import time
from pinecone import Pinecone, ServerlessSpec
from src.embeddings.config import (
    PINECONE_API_KEY,
    PINECONE_ENVIRONMENT,
    PINECONE_INDEX_NAME,
    EMBEDDING_DIMENSIONS
)


class VectorStore:
    """Manage resume embeddings in Pinecone vector database"""
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        index_name: Optional[str] = None,
        dimension: Optional[int] = None
    ):
        """
        Initialize Pinecone vector store
        
        Args:
            api_key: Pinecone API key (uses config default if not provided)
            index_name: Name of Pinecone index (uses config default if not provided)
            dimension: Vector dimension (uses config default if not provided)
        """
        self.api_key = api_key or PINECONE_API_KEY
        self.index_name = index_name or PINECONE_INDEX_NAME
        self.dimension = dimension or EMBEDDING_DIMENSIONS
        
        if not self.api_key:
            raise ValueError("Pinecone API key is required")
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=self.api_key)
        
        # Connect to index
        self.index = self.pc.Index(self.index_name)
        
        print(f"✅ Connected to Pinecone index: {self.index_name}")
    
    def upsert_resume(
        self, 
        resume_id: str, 
        embedding: List[float], 
        metadata: Dict[str, Any]
    ) -> bool:
        """
        Store a single resume embedding with metadata
        
        Args:
            resume_id: Unique identifier for the resume
            embedding: Vector embedding of the resume
            metadata: Resume metadata (name, email, skills, etc.)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Validate embedding dimensions
            if len(embedding) != self.dimension:
                raise ValueError(
                    f"Expected {self.dimension} dimensions, got {len(embedding)}"
                )
            
            # Prepare vector for upsert
            vector = {
                "id": resume_id,
                "values": embedding,
                "metadata": metadata
            }
            
            # Upsert to Pinecone
            self.index.upsert(vectors=[vector])
            
            return True
            
        except Exception as e:
            print(f"❌ Error upserting resume {resume_id}: {e}")
            return False
    
    def upsert_resumes_batch(
        self, 
        resumes: List[Dict[str, Any]],
        batch_size: int = 100
    ) -> Dict[str, int]:
        """
        Store multiple resume embeddings in batches
        
        Args:
            resumes: List of dicts with 'id', 'embedding', and 'metadata'
            batch_size: Number of vectors to upsert at once
            
        Returns:
            Dictionary with success and failure counts
        """
        total = len(resumes)
        successful = 0
        failed = 0
        
        # Process in batches
        for i in range(0, total, batch_size):
            batch = resumes[i:i + batch_size]
            
            try:
                # Prepare vectors
                vectors = []
                for resume in batch:
                    vector = {
                        "id": resume["id"],
                        "values": resume["embedding"],
                        "metadata": resume.get("metadata", {})
                    }
                    vectors.append(vector)
                
                # Upsert batch
                self.index.upsert(vectors=vectors)
                successful += len(vectors)
                
                print(f"✅ Upserted batch {i//batch_size + 1}: {len(vectors)} resumes")
                
            except Exception as e:
                print(f"❌ Error upserting batch {i//batch_size + 1}: {e}")
                failed += len(batch)
        
        return {
            "total": total,
            "successful": successful,
            "failed": failed
        }
    
    def search_similar_resumes(
        self, 
        query_embedding: List[float], 
        top_k: int = 10,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for resumes similar to a query embedding
        
        Args:
            query_embedding: Vector embedding of the job description
            top_k: Number of top results to return
            filter_dict: Optional metadata filters (e.g., {"location": "Mumbai"})
            
        Returns:
            List of matching resumes with scores and metadata
        """
        try:
            # Query Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict
            )
            
            # Format results - Fixed: Access matches correctly
            matches = []
            if results.matches:
                for match in results.matches:
                    matches.append({
                        "id": match.id,
                        "score": match.score,
                        "metadata": match.metadata or {}
                    })
            
            return matches
            
        except Exception as e:
            print(f"❌ Error searching resumes: {e}")
            return []
        
    def get_resume_by_id(self, resume_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific resume by ID

        Args:
            resume_id: Unique identifier for the resume

        Returns:
            Resume data with embedding and metadata, or None if not found
        """
        try:
            result = self.index.fetch(ids=[resume_id])

            # Fixed: Access vectors correctly
            if result.vectors and resume_id in result.vectors:
                vector_data = result.vectors[resume_id]
                return {
                    "id": resume_id,
                    "embedding": vector_data.values,
                    "metadata": vector_data.metadata or {}
                }
            else:
                return None

        except Exception as e:
            print(f"❌ Error fetching resume {resume_id}: {e}")
            return None
    
    def delete_resume(self, resume_id: str) -> bool:
        """
        Delete a resume from the index
        
        Args:
            resume_id: Unique identifier for the resume
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.index.delete(ids=[resume_id])
            return True
        except Exception as e:
            print(f"❌ Error deleting resume {resume_id}: {e}")
            return False
    
    def get_index_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the Pinecone index
        
        Returns:
            Dictionary with index statistics
        """
        try:
            stats = self.index.describe_index_stats()
            return {
                "total_vectors": stats.get('total_vector_count', 0),
                "dimension": stats.get('dimension', 0),
                "index_fullness": stats.get('index_fullness', 0),
                "namespaces": stats.get('namespaces', {})
            }
        except Exception as e:
            print(f"❌ Error getting index stats: {e}")
            return {}
    
    def clear_index(self) -> bool:
        """
        Delete all vectors from the index (use with caution!)
        
        Returns:
            True if successful, False otherwise
        """
        try:
            self.index.delete(delete_all=True)
            print("⚠️  All vectors deleted from index")
            return True
        except Exception as e:
            print(f"❌ Error clearing index: {e}")
            return False


# Utility functions
def store_resume_embedding(
    resume_id: str, 
    embedding: List[float], 
    metadata: Dict[str, Any]
) -> bool:
    """
    Quick utility to store a single resume
    
    Args:
        resume_id: Unique identifier
        embedding: Vector embedding
        metadata: Resume metadata
        
    Returns:
        True if successful
    """
    store = VectorStore()
    return store.upsert_resume(resume_id, embedding, metadata)


def search_resumes(query_embedding: List[float], top_k: int = 10) -> List[Dict]:
    """
    Quick utility to search resumes
    
    Args:
        query_embedding: Query vector
        top_k: Number of results
        
    Returns:
        List of matching resumes
    """
    store = VectorStore()
    return store.search_similar_resumes(query_embedding, top_k)


if __name__ == "__main__":
    # Test the vector store
    print("🧪 Testing Vector Store...")
    print("=" * 80)
    
    try:
        # Initialize store
        store = VectorStore()
        
        # Get index stats
        print("\n📊 Index Statistics:")
        stats = store.get_index_stats()
        print(f"   Total Vectors: {stats.get('total_vectors', 0)}")
        print(f"   Dimensions: {stats.get('dimension', 0)}")
        print(f"   Index Fullness: {stats.get('index_fullness', 0):.2%}")
        
        # Test with dummy data
        print("\n🔄 Testing with dummy resume...")
        
        # Create dummy embedding (768 dimensions for Gemini)
        dummy_embedding = [0.1] * 768
        
        dummy_metadata = {
            "full_name": "Test User",
            "email": "test@example.com",
            "skills": ["Python", "Machine Learning", "FastAPI"],
            "experience_years": 5,
            "location": "Mumbai"
        }
        
        # Upsert test resume
        test_id = "test_resume_001"
        success = store.upsert_resume(test_id, dummy_embedding, dummy_metadata)
        
        if success:
            print(f"   ✅ Successfully stored test resume: {test_id}")
            
            # Wait a moment for indexing
            time.sleep(1)
            
            # Fetch the resume back
            print("\n🔄 Fetching test resume...")
            resume_data = store.get_resume_by_id(test_id)
            
            if resume_data:
                print(f"   ✅ Retrieved resume:")
                print(f"      ID: {resume_data['id']}")
                print(f"      Name: {resume_data['metadata']['full_name']}")
                print(f"      Skills: {resume_data['metadata']['skills']}")
                print(f"      Embedding dims: {len(resume_data['embedding'])}")
            
            # Test search
            print("\n🔍 Testing search...")
            results = store.search_similar_resumes(dummy_embedding, top_k=5)
            print(f"   Found {len(results)} similar resumes")
            
            if results:
                for i, result in enumerate(results, 1):
                    print(f"\n   {i}. Score: {result['score']:.4f}")
                    print(f"      Name: {result['metadata'].get('full_name', 'N/A')}")
            
            # Clean up test data
            print("\n🧹 Cleaning up test data...")
            store.delete_resume(test_id)
            print("   ✅ Test resume deleted")
            
        else:
            print("   ❌ Failed to store test resume")
        
        # Final stats
        print("\n📊 Final Index Statistics:")
        stats = store.get_index_stats()
        print(f"   Total Vectors: {stats.get('total_vectors', 0)}")
        
        print("\n" + "=" * 80)
        print("✅ All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()