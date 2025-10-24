"""
test_job_match.py

Test the job-to-candidate matching functionality:
1. Take a job description
2. Convert to embedding
3. Search Pinecone for similar candidates
4. Display top matches with scores

This tests Stage 1 of the recommendation engine (vector similarity search).
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.embeddings.embedding_generator import EmbeddingGenerator
from src.embeddings.preprocessor import ResumePreprocessor
from src.embeddings.vector_store import VectorStore
import json
from datetime import datetime
from typing import List


class JobMatcher:
    """Test job description matching against resume database"""
    
    def __init__(self):
        """Initialize components"""
        print("🔄 Initializing job matcher...")
        self.embedding_generator = EmbeddingGenerator()
        self.preprocessor = ResumePreprocessor()
        self.vector_store = VectorStore()
        print("✅ Job matcher initialized\n")
    
    def preprocess_job_description(self, jd_text: str) -> str:
        """
        Preprocess job description for embedding
        
        Args:
            jd_text: Raw job description text
            
        Returns:
            Preprocessed text ready for embedding
        """
        # For JD, we just clean and truncate if needed
        # JDs are usually shorter than resumes
        
        # Remove extra whitespace
        jd_text = " ".join(jd_text.split())
        
        # Truncate if too long (keep under 2000 tokens ~ 8000 chars)
        max_length = 8000
        if len(jd_text) > max_length:
            jd_text = jd_text[:max_length] + "..."
            print(f"   ⚠️  JD truncated to {max_length} chars")
        
        return jd_text
    
    def aggregate_chunk_scores(self, matches: List) -> List:
        """Combine scores from same resume's chunks"""
        from collections import defaultdict

        resume_scores = defaultdict(lambda: {"scores": [], "metadata": None})

        for match in matches:
            # Extract base resume_id (remove _chunk0, _chunk1, etc.)
            base_id = match['id'].rsplit('_chunk', 1)[0]
            resume_scores[base_id]["scores"].append(match['score'])
            if resume_scores[base_id]["metadata"] is None:
                resume_scores[base_id]["metadata"] = match['metadata']

        # Take MAX score per resume (best matching chunk)
        aggregated = []
        for resume_id, data in resume_scores.items():
            aggregated.append({
                "id": resume_id,
                "score": sum(data["scores"]) / len(data["scores"]),  # Average
                "metadata": data["metadata"]
            })

        # Re-sort by score
        aggregated.sort(key=lambda x: x['score'], reverse=True)
        return aggregated

    def search_candidates(
        self, 
        job_description: str, 
        top_k: int = 10,
        filter_dict: dict = None
    ):
        """
        Search for candidates matching a job description
        
        Args:
            job_description: The job description text
            top_k: Number of top candidates to return
            filter_dict: Optional filters (e.g., location, skills)
            
        Returns:
            List of matching candidates with scores
        """
        print("=" * 80)
        print("🔍 JOB MATCHING TEST")
        print("=" * 80)
        
        # Step 1: Preprocess JD
        print("\n📝 Step 1: Preprocessing job description...")
        preprocessed_jd = self.preprocess_job_description(job_description)
        token_estimate = len(preprocessed_jd.split())
        print(f"   ✅ JD length: {len(preprocessed_jd)} chars")
        print(f"   ✅ Token estimate: ~{token_estimate}")
        
        # Step 2: Generate embedding
        print("\n🧬 Step 2: Generating JD embedding...")
        jd_embedding = self.embedding_generator.generate_embedding(preprocessed_jd)
        print(f"   ✅ Embedding dims: {len(jd_embedding)}")
        
        # Step 3: Search vector database
        print(f"\n🔍 Step 3: Searching for top {top_k} candidates...")
        matches = self.vector_store.search_similar_resumes(
            query_embedding=jd_embedding,
            top_k=top_k,
            filter_dict=filter_dict
        )

        # Aggregate chunks if any
        matches = self.aggregate_chunk_scores(matches)

        
        if not matches:
            print("   ❌ No matches found!")
            return []
        
        print(f"   ✅ Found {len(matches)} candidates\n")
        
        # Step 4: Display results
        print("=" * 80)
        print("📊 TOP MATCHING CANDIDATES")
        print("=" * 80)
        
        for i, match in enumerate(matches, 1):
            score = match['score']
            metadata = match['metadata']
            
            print(f"\n🏆 Rank #{i} - Score: {score:.4f} ({score*100:.1f}% match)")
            print(f"   👤 Name: {metadata.get('full_name', 'N/A')}")
            print(f"   📧 Email: {metadata.get('email', 'N/A')}")
            print(f"   💼 Title: {metadata.get('professional_title', 'N/A')}")
            print(f"   📍 Location: {metadata.get('location', 'N/A')}")
            print(f"   🎯 Skills: {metadata.get('total_skills', 0)} total")
            
            # Show top skills
            skills = metadata.get('skills', [])
            if skills:
                print(f"   🔧 Top Skills: {', '.join(skills[:5])}")
            
            print(f"   💼 Experience: {metadata.get('experience_count', 0)} jobs")
            print(f"   🎓 Education: {metadata.get('education_count', 0)} degrees")
        
        print("\n" + "=" * 80)
        
        # Save results
        self.save_results(job_description, matches)
        
        return matches
    
    def save_results(self, job_description: str, matches: list):
        """Save search results to JSON"""
        import os
        os.makedirs("../outputs/job_matches", exist_ok=True)
        
        result_data = {
            "timestamp": datetime.now().isoformat(),
            "job_description": job_description[:500] + "..." if len(job_description) > 500 else job_description,
            "total_matches": len(matches),
            "matches": [
                {
                    "rank": i + 1,
                    "score": match['score'],
                    "candidate": {
                        "name": match['metadata'].get('full_name'),
                        "email": match['metadata'].get('email'),
                        "title": match['metadata'].get('professional_title'),
                        "location": match['metadata'].get('location'),
                        "skills_count": match['metadata'].get('total_skills'),
                        "top_skills": match['metadata'].get('skills', [])[:10]
                    }
                }
                for i, match in enumerate(matches)
            ]
        }
        
        filename = f"outputs/job_matches/match_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(result_data, f, indent=2)
        
        print(f"💾 Results saved to: {filename}")


def test_with_sample_jd():
    """Test with a sample job description"""
    
    # Sample Job Description (AI/ML Engineer)
    job_description = """
    Senior AI/ML Engineer
    
    We are seeking an experienced AI/ML Engineer to join our team. The ideal candidate 
    will have strong expertise in machine learning, deep learning, and natural language 
    processing.
    
    Required Skills:
    - Python, PyTorch, TensorFlow
    - Machine Learning and Deep Learning
    - Natural Language Processing (NLP)
    - LLMs and Generative AI
    - Experience with RAG systems
    - Cloud platforms (AWS/GCP)
    - Docker and MLOps
    
    Responsibilities:
    - Design and implement ML models
    - Build and deploy AI systems
    - Work with large language models
    - Optimize model performance
    - Collaborate with cross-functional teams
    
    Requirements:
    - 3+ years of ML/AI experience
    - Bachelor's degree in Computer Science or related field
    - Strong programming skills in Python
    - Experience with production ML systems
    """
    
    print("\n" + "🎯" * 40)
    print("TESTING JOB MATCHING SYSTEM")
    print("🎯" * 40)
    
    print("\n📋 Job Description:")
    print("-" * 80)
    print(job_description.strip())
    print("-" * 80)
    
    # Create matcher and search
    matcher = JobMatcher()
    
    # Search for top 10 candidates
    matches = matcher.search_candidates(
        job_description=job_description,
        top_k=10
    )
    
    # Summary
    print("\n" + "=" * 80)
    print("✅ JOB MATCHING TEST COMPLETE")
    print("=" * 80)
    print(f"Total candidates found: {len(matches)}")
    
    if matches:
        print(f"Best match: {matches[0]['metadata'].get('full_name')} ({matches[0]['score']:.4f})")
        print(f"Worst match: {matches[-1]['metadata'].get('full_name')} ({matches[-1]['score']:.4f})")


def test_with_custom_jd():
    """Test with user-provided job description"""
    
    print("\n" + "🎯" * 40)
    print("CUSTOM JOB DESCRIPTION TEST")
    print("🎯" * 40)
    
    print("\nEnter your job description (press Ctrl+D or Ctrl+Z when done):")
    print("-" * 80)
    
    lines = []
    try:
        while True:
            line = input()
            lines.append(line)
    except EOFError:
        pass
    
    job_description = "\n".join(lines)
    
    if not job_description.strip():
        print("❌ No job description provided!")
        return
    
    # Create matcher and search
    matcher = JobMatcher()
    
    # Ask for number of results
    print("\nHow many top candidates to show? (default: 10)")
    try:
        top_k = int(input("Enter number: ").strip() or "10")
    except:
        top_k = 10
    
    matches = matcher.search_candidates(
        job_description=job_description,
        top_k=top_k
    )
    
    print(f"\n✅ Found {len(matches)} matching candidates!")


def main():
    """Main test execution"""
    
    print("\n" + "=" * 80)
    print("JOB MATCHING TEST SUITE")
    print("=" * 80)
    
    print("\nSelect test mode:")
    print("1. Test with sample AI/ML job description")
    print("2. Test with custom job description")
    print("3. Run both tests")
    
    choice = input("\nEnter choice (1/2/3): ").strip()
    
    if choice == "1":
        test_with_sample_jd()
    elif choice == "2":
        test_with_custom_jd()
    elif choice == "3":
        test_with_sample_jd()
        print("\n" + "=" * 80 + "\n")
        test_with_custom_jd()
    else:
        print("Invalid choice. Running sample test...")
        test_with_sample_jd()


if __name__ == "__main__":
    main()