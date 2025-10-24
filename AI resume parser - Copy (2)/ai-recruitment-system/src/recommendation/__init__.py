"""
Recommendation Module - Two-Stage Candidate Recommendation System

Stage 1: Fast vector-based semantic search (recommendation_engine)
Stage 2: LLM-powered re-ranking with explanations (llm_reranker)
"""

from .recommendation_engine import (
    RecommendationEngine,
    CandidateMatch
)

from .llm_reranker import (
    LLMReranker,
    RankedCandidate,
    RerankerOutput
)

_all_ = [
    'RecommendationEngine',
    'CandidateMatch',
    'LLMReranker',
    'RankedCandidate',
    'RerankerOutput'
]

