"""
LLM Re-ranker - Stage 2: Intelligent Candidate Ranking with Explanations
Uses K2 Think/Gemini to re-rank candidates with detailed reasoning
"""

import os
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

load_dotenv()


class RankedCandidate(BaseModel):
    """Ranked candidate with explanation"""
    candidate_id: str = Field(description="Candidate ID")
    rank: int = Field(description="Final rank (1-10)")
    overall_score: float = Field(description="Overall match score 0-1")
    
    # Scoring breakdown
    skills_match: float = Field(description="Skills match score 0-1")
    experience_match: float = Field(description="Experience match score 0-1")
    location_match: float = Field(description="Location match score 0-1")
    
    # Explanation
    rationale: str = Field(description="2-3 sentence explanation of why this candidate matches")
    strengths: List[str] = Field(description="Key strengths (3-5 points)")
    concerns: Optional[List[str]] = Field(default=None, description="Potential concerns or gaps")
    
    # Criteria mapping
    criteria_met: Dict[str, bool] = Field(description="Which job requirements are met")
    missing_skills: Optional[List[str]] = Field(default=None, description="Skills mentioned in JD but not in resume")


class RerankerOutput(BaseModel):
    """Complete re-ranking output"""
    ranked_candidates: List[RankedCandidate] = Field(description="Top 10 ranked candidates")
    processing_time: float = Field(description="Time taken in seconds")
    model_used: str = Field(description="LLM model used")


class LLMReranker:
    """
    Stage 2 Re-ranker using LLM for intelligent ranking
    
    Takes top-K candidates from vector search and re-ranks them using:
    - Detailed job description analysis
    - Explicit criteria mapping
    - Natural language explanations
    - Best-of-N sampling for quality
    """
    
    def _init_(
        self,
        model_name: str = "gemini-2.0-flash-exp",
        temperature: float = 0.1,
        max_output_tokens: int = 8192
    ):
        """
        Initialize LLM reranker
        
        Args:
            model_name: LLM model to use (gemini-2.0-flash-exp, gpt-4o, etc.)
            temperature: Sampling temperature (lower = more deterministic)
            max_output_tokens: Maximum tokens in response
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        
        # Initialize LLM
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment")
        
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            google_api_key=api_key,
            max_output_tokens=max_output_tokens
        )
        
        # Setup prompt template
        self._setup_prompt()
        
        print(f"✅ LLM Reranker initialized")
        print(f"   Model: {model_name}")
        print(f"   Temperature: {temperature}")
    
    def _setup_prompt(self):
        """Setup re-ranking prompt template"""
        self.rerank_template = PromptTemplate(
            template="""You are an expert recruiter. Re-rank these candidates for a job and provide detailed explanations.

JOB DESCRIPTION:
{job_description}

KEY REQUIREMENTS:
{requirements}

CANDIDATES TO RANK (from initial search):
{candidates}

TASK:
Analyze each candidate and rank the top 10 based on:
1. Skills match (technical skills, tools, frameworks)
2. Experience level and relevance
3. Location fit
4. Overall suitability for the role

For EACH of the top 10 candidates, provide:
- Rank (1-10, where 1 is best match)
- Overall score (0-1)
- Score breakdown: skills_match, experience_match, location_match (each 0-1)
- Rationale: 2-3 sentences explaining WHY this candidate matches
- Strengths: 3-5 key strengths
- Concerns: Any gaps or concerns (or null if none)
- Criteria met: Map each requirement to true/false
- Missing skills: Skills in JD but not in resume

CRITICAL RULES:
- Be objective and fair
- Base scores on evidence from resume
- Explain your reasoning clearly
- Highlight both strengths AND concerns
- Return ONLY valid JSON

Return JSON in this exact format:
{{
  "ranked_candidates": [
    {{
      "candidate_id": "string",
      "rank": 1,
      "overall_score": 0.95,
      "skills_match": 0.9,
      "experience_match": 1.0,
      "location_match": 0.95,
      "rationale": "Candidate has 6 years of React experience with AWS, exceeding the 5-year requirement. Strong background in microservices architecture aligns perfectly with the role.",
      "strengths": ["6+ years React & AWS", "Microservices expertise", "Led team of 5", "Bangalore-based", "Immediate availability"],
      "concerns": ["No GraphQL experience mentioned", "Limited mobile development"],
      "criteria_met": {{
        "5+ years React": true,
        "AWS experience": true,
        "Team leadership": true,
        "GraphQL": false,
        "Bangalore location": true
      }},
      "missing_skills": ["GraphQL", "React Native"]
    }}
  ]
}}

Return ONLY the JSON, no extra text.""",
            input_variables=["job_description", "requirements", "candidates"]
        )
    
    def rerank_candidates(
        self,
        candidates: List[Dict[str, Any]],
        job_description: str,
        requirements: Optional[List[str]] = None,
        top_n: int = 10,
        use_best_of_n: bool = False,
        n_samples: int = 3
    ) -> RerankerOutput:
        """
        Re-rank candidates using LLM
        
        Args:
            candidates: List of candidate dicts from Stage 1
            job_description: Full job description text
            requirements: Explicit list of requirements (optional)
            top_n: Number of top candidates to return
            use_best_of_n: Use Best-of-N sampling for quality
            n_samples: Number of samples for Best-of-N
        
        Returns:
            RerankerOutput with ranked candidates and explanations
        """
        start_time = datetime.now()
        
        print(f"\n🤖 Re-ranking {len(candidates)} candidates...")
        print(f"   Model: {self.model_name}")
        if use_best_of_n:
            print(f"   Using Best-of-{n_samples} sampling")
        
        # Prepare requirements
        if not requirements:
            requirements = self._extract_requirements(job_description)
        
        requirements_text = "\n".join([f"- {req}" for req in requirements])
        
        # Prepare candidates text
        candidates_text = self._format_candidates(candidates)
        
        # Create prompt
        prompt = self.rerank_template.format(
            job_description=job_description,
            requirements=requirements_text,
            candidates=candidates_text
        )
        
        try:
            if use_best_of_n:
                # Best-of-N: Generate multiple responses, select best
                response = self._best_of_n_rerank(prompt, n_samples)
            else:
                # Standard: Single generation
                response = self.llm.invoke(prompt)
                response_text = response.content.strip()
                response = self._parse_response(response_text)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Validate and limit to top_n
            ranked_candidates = response.get('ranked_candidates', [])[:top_n]
            
            print(f"✅ Re-ranking complete in {processing_time:.2f}s")
            print(f"   Ranked {len(ranked_candidates)} candidates")
            
            return RerankerOutput(
                ranked_candidates=[RankedCandidate(**c) for c in ranked_candidates],
                processing_time=processing_time,
                model_used=self.model_name
            )
        
        except Exception as e:
            print(f"❌ Error in re-ranking: {e}")
            raise
    
    def _best_of_n_rerank(self, prompt: str, n: int) -> Dict[str, Any]:
        """
        Best-of-N sampling: Generate N responses, select best
        
        Quality metric: Average overall_score of top 3 candidates
        """
        print(f"   Generating {n} candidate rankings...")
        
        responses = []
        for i in range(n):
            try:
                response = self.llm.invoke(prompt)
                parsed = self._parse_response(response.content.strip())
                
                # Calculate quality score (avg of top 3)
                top_3_scores = [
                    c['overall_score'] 
                    for c in parsed.get('ranked_candidates', [])[:3]
                ]
                quality_score = sum(top_3_scores) / len(top_3_scores) if top_3_scores else 0
                
                responses.append({
                    'response': parsed,
                    'quality_score': quality_score
                })
                
                print(f"      Sample {i+1}: Quality = {quality_score:.3f}")
            
            except Exception as e:
                print(f"      Sample {i+1}: Failed - {e}")
                continue
        
        if not responses:
            raise ValueError("All Best-of-N samples failed")
        
        # Select best response
        best_response = max(responses, key=lambda x: x['quality_score'])
        print(f"   ✅ Selected best sample (quality: {best_response['quality_score']:.3f})")
        
        return best_response['response']
    
    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """Parse LLM response to JSON"""
        # Clean response
        if response_text.startswith(''):
            response_text = response_text.split('')[1]
            if response_text.startswith('json'):
                response_text = response_text[4:]
            response_text = response_text.strip()
        
        # Parse JSON
        try:
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            print(f"❌ JSON parse error: {e}")
            print(f"Response preview: {response_text[:500]}")
            raise
    
    def _format_candidates(self, candidates: List[Dict[str, Any]]) -> str:
        """Format candidates for prompt"""
        formatted = []
        
        for i, candidate in enumerate(candidates, 1):
            skills_text = ", ".join(candidate.get('skills', [])[:10])
            
            candidate_text = f"""
Candidate {i}:
- ID: {candidate.get('candidate_id', 'unknown')}
- Name: {candidate.get('name', 'Unknown')}
- Title: {candidate.get('professional_title', 'N/A')}
- Experience: {candidate.get('experience_years', 0)} years
- Location: {candidate.get('location', 'N/A')}
- Skills: {skills_text}
- Vector Score: {candidate.get('score', 0):.3f}
"""
            formatted.append(candidate_text.strip())
        
        return "\n\n".join(formatted)
    
    def _extract_requirements(self, job_description: str) -> List[str]:
        """Extract key requirements from job description"""
        # Simple extraction - can be improved with LLM
        requirements = []
        
        keywords = ['required', 'must have', 'experience with', 'proficiency in']
        lines = job_description.lower().split('\n')
        
        for line in lines:
            if any(kw in line for kw in keywords):
                requirements.append(line.strip('- •').strip())
        
        # If no requirements found, return generic
        if not requirements:
            requirements = [
                "Relevant technical skills",
                "Sufficient experience level",
                "Location compatibility"
            ]
        
        return requirements
    
    def explain_ranking(self, candidate: RankedCandidate) -> str:
        """Generate human-readable explanation"""
        explanation = f"""
🏆 Rank #{candidate.rank}

👤 Candidate ID: {candidate.candidate_id}
📊 Overall Score: {candidate.overall_score:.2%}

📈 Score Breakdown:
   • Skills Match: {candidate.skills_match:.2%}
   • Experience Match: {candidate.experience_match:.2%}
   • Location Match: {candidate.location_match:.2%}

💡 Rationale:
{candidate.rationale}

✅ Key Strengths:
{chr(10).join(['   • ' + s for s in candidate.strengths])}
"""
        
        if candidate.concerns:
            explanation += f"\n⚠  Potential Concerns:\n"
            explanation += "\n".join(['   • ' + c for c in candidate.concerns])
        
        if candidate.missing_skills:
            explanation += f"\n\n📝 Missing Skills:\n"
            explanation += ", ".join(candidate.missing_skills)
        
        return explanation


# Example usage
if _name_ == "_main_":
    # Initialize reranker
    reranker = LLMReranker()
    
    # Example candidates (from Stage 1)
    candidates = [
        {
            'candidate_id': '123',
            'name': 'John Doe',
            'professional_title': 'Senior Full Stack Developer',
            'experience_years': 6,
            'location': 'Bangalore',
            'skills': ['React', 'Node.js', 'AWS', 'MongoDB', 'Docker'],
            'score': 0.89
        },
        {
            'candidate_id': '456',
            'name': 'Jane Smith',
            'professional_title': 'Frontend Developer',
            'experience_years': 4,
            'location': 'Mumbai',
            'skills': ['React', 'TypeScript', 'Redux', 'CSS', 'Jest'],
            'score': 0.85
        }
    ]
    
    # Job description
    job_desc = """
    Senior Full Stack Developer
    
    Requirements:
    - 5+ years of experience with React and Node.js
    - Strong AWS experience
    - Experience with microservices architecture
    - Based in Bangalore or willing to relocate
    - Team leadership experience
    """
    
    # Re-rank
    result = reranker.rerank_candidates(
        candidates=candidates,
        job_description=job_desc,
        top_n=2,
        use_best_of_n=False
    )
    
    # Print results
    print("\n" + "="*60)
    print("RE-RANKING RESULTS")
    print("="*60)
    
    for candidate in result.ranked_candidates:
        print(reranker.explain_ranking(candidate))
        print("-"*60)