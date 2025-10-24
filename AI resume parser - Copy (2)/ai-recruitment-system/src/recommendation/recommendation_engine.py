"""
Recommendation Engine - Stage 1: Vector-based Candidate Retrieval
Performs fast semantic search using embeddings to find top-N candidates
"""

import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from dotenv import load_dotenv

try:
    from pinecone import Pinecone, ServerlessSpec
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    print("⚠  Pinecone not installed. Install with: pip install pinecone-client")

load_dotenv()


@dataclass
class CandidateMatch:
    """Represents a candidate match from vector search"""
    candidate_id: str
    score: float
    name: str
    email: str
    skills: List[str]
    experience_years: float
    location: Optional[str]
    professional_title: Optional[str]
    metadata: Dict[str, Any]


class RecommendationEngine:
    """
    Stage 1 Recommendation Engine using Vector Similarity Search
    
    Fast semantic matching to retrieve top-N candidates from large database
    Uses cosine similarity on resume embeddings stored in Pinecone
    """
    
    def _init_(
        self, 
        index_name: str = "resume-embeddings",
        dimension: int = 768,  # text-embedding-004 dimension
        metric: str = "cosine"
    ):
        """
        Initialize recommendation engine
        
        Args:
            index_name: Name of Pinecone index
            dimension: Embedding dimension (768 for text-embedding-004)
            metric: Distance metric (cosine, euclidean, dotproduct)
        """
        if not PINECONE_AVAILABLE:
            raise ImportError("Pinecone client not available")
        
        self.index_name = index_name
        self.dimension = dimension
        self.metric = metric
        
        # Initialize Pinecone
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY not found in environment")
        
        self.pc = Pinecone(api_key=api_key)
        
        # Connect to or create index
        self._setup_index()
        
        print(f"✅ Recommendation Engine initialized")
        print(f"   Index: {self.index_name}")
        print(f"   Dimension: {self.dimension}")
        print(f"   Metric: {self.metric}")
    
    def _setup_index(self):
        """Setup or connect to Pinecone index"""
        try:
            # Check if index exists
            existing_indexes = [idx.name for idx in self.pc.list_indexes()]
            
            if self.index_name not in existing_indexes:
                print(f"⚠  Index '{self.index_name}' not found. Creating new index...")
                
                # Create new index
                self.pc.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric=self.metric,
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"  # Change based on your region
                    )
                )
                print(f"✅ Created new index: {self.index_name}")
            
            # Connect to index
            self.index = self.pc.Index(self.index_name)
            
            # Get index stats
            stats = self.index.describe_index_stats()
            print(f"📊 Index stats: {stats.total_vector_count} vectors")
            
        except Exception as e:
            print(f"❌ Error setting up index: {e}")
            raise
    
    def search_candidates(
        self,
        query_embedding: List[float],
        top_k: int = 50,
        filters: Optional[Dict[str, Any]] = None,
        min_score: float = 0.0
    ) -> List[CandidateMatch]:
        """
        Search for candidates using query embedding
        
        Args:
            query_embedding: Job description embedding vector
            top_k: Number of candidates to retrieve
            filters: Metadata filters (e.g., location, experience)
            min_score: Minimum similarity score threshold
        
        Returns:
            List of CandidateMatch objects sorted by relevance
        """
        try:
            print(f"\n🔍 Searching for top {top_k} candidates...")
            
            # Build filter dict for Pinecone
            pinecone_filter = self._build_filter(filters) if filters else None
            
            # Query Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=pinecone_filter
            )
            
            # Parse results
            matches = []
            for match in results.matches:
                # Filter by minimum score
                if match.score < min_score:
                    continue
                
                metadata = match.metadata
                
                candidate = CandidateMatch(
                    candidate_id=match.id,
                    score=float(match.score),
                    name=metadata.get('name', 'Unknown'),
                    email=metadata.get('email', 'N/A'),
                    skills=metadata.get('skills', []),
                    experience_years=float(metadata.get('experience_years', 0)),
                    location=metadata.get('location'),
                    professional_title=metadata.get('professional_title'),
                    metadata=metadata
                )
                
                matches.append(candidate)
            
            print(f"✅ Found {len(matches)} candidates")
            if matches:
                print(f"   Top score: {matches[0].score:.3f}")
                print(f"   Top candidate: {matches[0].name}")
            
            return matches
        
        except Exception as e:
            print(f"❌ Error searching candidates: {e}")
            return []
    
    def _build_filter(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build Pinecone filter from search criteria
        
        Supported filters:
        - location: str or List[str]
        - min_experience: float
        - max_experience: float
        - required_skills: List[str]
        - education_level: str
        """
        pinecone_filter = {}
        
        # Location filter
        if 'location' in filters:
            location = filters['location']
            if isinstance(location, list):
                pinecone_filter['location'] = {"$in": location}
            else:
                pinecone_filter['location'] = location
        
        # Experience filters
        if 'min_experience' in filters or 'max_experience' in filters:
            exp_filter = {}
            if 'min_experience' in filters:
                exp_filter['$gte'] = filters['min_experience']
            if 'max_experience' in filters:
                exp_filter['$lte'] = filters['max_experience']
            pinecone_filter['experience_years'] = exp_filter
        
        # Skills filter (candidate must have at least one)
        if 'required_skills' in filters:
            skills = filters['required_skills']
            if skills:
                # Pinecone doesn't support array contains well, 
                # so we handle this in post-filtering
                pass
        
        return pinecone_filter if pinecone_filter else None
    
    def post_filter_by_skills(
        self,
        candidates: List[CandidateMatch],
        required_skills: List[str],
        min_match_count: int = 1
    ) -> List[CandidateMatch]:
        """
        Post-filter candidates by required skills
        
        Args:
            candidates: List of candidates from vector search
            required_skills: List of required skill keywords
            min_match_count: Minimum number of skills that must match
        
        Returns:
            Filtered list of candidates
        """
        if not required_skills:
            return candidates
        
        filtered = []
        required_skills_lower = [s.lower() for s in required_skills]
        
        for candidate in candidates:
            candidate_skills_lower = [s.lower() for s in candidate.skills]
            
            # Count matching skills
            match_count = sum(
                1 for req_skill in required_skills_lower
                if any(req_skill in cand_skill for cand_skill in candidate_skills_lower)
            )
            
            if match_count >= min_match_count:
                # Add match count to metadata
                candidate.metadata['skill_match_count'] = match_count
                filtered.append(candidate)
        
        # Re-sort by skill match count then by score
        filtered.sort(key=lambda x: (x.metadata.get('skill_match_count', 0), x.score), reverse=True)
        
        return filtered
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector index"""
        try:
            stats = self.index.describe_index_stats()
            return {
                'total_vectors': stats.total_vector_count,
                'dimension': stats.dimension,
                'index_fullness': stats.index_fullness,
                'namespaces': stats.namespaces
            }
        except Exception as e:
            print(f"❌ Error getting index stats: {e}")
            return {}
    
    def delete_candidate(self, candidate_id: str):
        """Delete a candidate from the index"""
        try:
            self.index.delete(ids=[candidate_id])
            print(f"✅ Deleted candidate: {candidate_id}")
        except Exception as e:
            print(f"❌ Error deleting candidate: {e}")


# Example usage
if _name_ == "_main_":
    # Initialize engine
    engine = RecommendationEngine()
    
    # Example: Search with dummy query embedding
    dummy_query = [0.1] * 768  # Replace with actual embedding
    
    # Search without filters
    candidates = engine.search_candidates(
        query_embedding=dummy_query,
        top_k=10,
        min_score=0.7
    )
    
    # Print results
    print("\n📋 Top Candidates:")
    for i, candidate in enumerate(candidates, 1):
        print(f"{i}. {candidate.name} - Score: {candidate.score:.3f}")
        print(f"   Title: {candidate.professional_title}")
        print(f"   Skills: {', '.join(candidate.skills[:5])}")
        print()
    
    # Example: Search with filters
    candidates_filtered = engine.search_candidates(
        query_embedding=dummy_query,
        top_k=50,
        filters={
            'location': 'Bangalore',
            'min_experience': 3,
            'max_experience': 8
        }
    )
    
    print(f"\n🔍 Filtered candidates: {len(candidates_filtered)}")
