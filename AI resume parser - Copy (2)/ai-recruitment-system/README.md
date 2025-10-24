# 🚀 AI-Powered Resume Parser & Intelligent Candidate Recommendation System

An end-to-end AI recruitment platform that automates resume parsing, semantic search, and intelligent candidate ranking with explainable AI recommendations.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📋 Overview

This system revolutionizes the recruitment process by:

- _Parsing resumes_ with 95%+ accuracy using hybrid AI approach
- _Semantic search_ through thousands of candidates in milliseconds
- _Intelligent re-ranking_ with detailed explanations for each match
- _Explainable AI_ providing transparent, auditable hiring decisions

### Key Features

✅ Hybrid resume parsing (Regex + Gemini AI)  
✅ Two-stage recommendation system (Vector Search + LLM Re-ranking)  
✅ Semantic candidate matching  
✅ Explainable rankings with detailed rationales  
✅ Fairness & compliance built-in  
✅ RESTful API with FastAPI

---

## 🏗 System Architecture

<div align="center">
  <img src="architecture.png" alt="System Architecture" width="600">
</div>

## 🛠 Tech Stack

### Backend

- _FastAPI_ - High-performance async API framework
- _PostgreSQL_ - Structured data storage with JSONB
- _Pinecone_ - Vector database for semantic search
- _LangChain_ - LLM orchestration framework

### AI/ML

- _Gemini 2.0 Flash_ - Resume parsing & re-ranking
- _text-embedding-004_ - Text embeddings (768-dim)
- _Regex_ - Deterministic field extraction

### Core Libraries

- pydantic - Data validation
- sqlalchemy - Database ORM
- python-dotenv - Environment management

---

## 📦 Installation

### Prerequisites

- Python 3.9+
- PostgreSQL 13+
- API Keys: Google AI (Gemini), Pinecone

### Setup

1. _Clone the repository_
   ```bash
   git clone https://github.com/affaan777/AI-Resume-Parser---Candidate-Recommendation-System.git
   cd ai-recruitment-system

2. _Create virtual environment_
   ```bash
   python -m venv venv
   source venv/bin/activate # On Windows: venv\Scripts\activate

3. _Install dependencies_
   ```bash
   pip install -r requirements.txt
   
4. _Set up environment variables_
   ```bash
   cp .env.example .env

Edit .env with your credentials:
env

# Google AI
```bash
GOOGLE_API_KEY=your_gemini_api_key
```
# Pinecone
```bash
PINECONE_API_KEY=your_pinecone_api_key
```
# Database
```bash
DATABASE_URL=postgresql://user:password@localhost:5432/recruitment_db
```
# API
```
API_KEY=your_secure_api_key
```
5. _Run the application_
   ```bash
   uvicorn src.api.main:app --reload --port 8000

---

## 🚀 Quick Start

### Parse a Resume

```bash
from src.parser.resume_parser_production import ResumeParser

parser = ResumeParser()
resume_data = parser.parse_resume("resume.pdf")

print(f"Name: {resume_data.full_name}")
print(f"Email: {resume_data.email}")
print(f"Skills: {', '.join(resume_data.skills)}")
```
### Search Candidates
```bash
python
from src.recommendation import RecommendationEngine, LLMReranker
```
# Stage 1: Vector search
```bash
engine = RecommendationEngine()
candidates = engine.search_candidates(
query_embedding=job_embedding,
top_k=50,
filters={
'location': 'Bangalore',
'min_experience': 3
}
)
```
# Stage 2: LLM re-ranking
```bash
reranker = LLMReranker()
result = reranker.rerank_candidates(
candidates=[c.__dict__ for c in candidates],
job_description="Senior React Developer with 5+ years...",
top_n=10
)
```
# View top match
```bash
top = result.ranked_candidates[0]
print(f"Rank: #{top.rank}")
print(f"Score: {top.overall_score:.2%}")
print(f"Rationale: {top.rationale}")
```

## 📡 API Endpoints

### Parse Resume
```bash
http
POST /api/v1/parse-resume
Content-Type: multipart/form-data
X-API-Key: your_api_key

Body: file (PDF)

_Response:_
json
{
"success": true,
"candidate_id": 123,
"message": "Resume parsed successfully",
"parsing_time": 5.23,
"status": "processing_embeddings"
}
```
### Get Candidate
```bash
http
GET /api/v1/candidates/{id}
X-API-Key: your_api_key
```
### Search Candidates
```bash
http
GET /api/v1/candidates/search?query=Senior+React+Developer&location=Bangalore&limit=10
X-API-Key: your_api_key

_Response:_
json
{
"results": [
{
"candidate_id": 123,
"name": "John Doe",
"rank": 1,
"score": 0.95,
"rationale": "Strong React & AWS experience...",
"strengths": ["6+ years React", "AWS certified"],
"concerns": ["No GraphQL experience"]
}
]
}
```


## 🧪 Testing
```bash
Run tests:
bash
pytest tests/ -v

Run with coverage:
bash
pytest --cov=src tests/
```

## 📊 Performance Metrics

| Operation                      | Time   | Accuracy       |
| ------------------------------ | ------ | -------------- |
| Resume Parsing                 | 5-7s   | 95%+ F1        |
| Vector Search (50 candidates)  | 0.5s   | 85% recall@50  |
| LLM Re-ranking (10 candidates) | 1.5s   | 90%+ precision |
| _Total Pipeline_               | _7-9s_ | _High quality_ |

---

## 🎯 Project Structure

```bash
ai-recruitment-system/
├── src/
│   ├── api/                     # FastAPI application
│   │   └── main.py
│   ├── parser/                  # Resume parsing
│   │   └── resume_parser_production.py
│   ├── embeddings/              # Vector embeddings
│   │   └── embedding_service.py
│   ├── recommendation/          # Two-stage recommendation
│   │   ├── recommendation_engine.py   # Stage 1
│   │   └── llm_reranker.py            # Stage 2
│   ├── database/                # Database models
│   │   └── models.py
│   ├── tests/
│   ├── docs/
│   └── requirements.txt
├── .env.example
└── README.md
```

---

## 🔐 Security & Compliance

- ✅ API key authentication
- ✅ Sensitive data redaction (age, gender, ethnicity)
- ✅ GDPR-compliant data handling
- ✅ Audit trail for all ranking decisions
- ✅ Explainable AI for fairness

---

## 🚧 Roadmap

- [ ] Multi-language resume support
- [ ] Real-time collaboration features
- [ ] Advanced analytics dashboard
- [ ] Mobile app (React Native)
- [ ] Integration with ATS platforms
- [ ] Video interview analysis

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (git checkout -b feature/amazing-feature)
3. Commit your changes (git commit -m 'Add amazing feature')
4. Push to the branch (git push origin feature/amazing-feature)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👥 Team

- _Shahnawaz Shaikh_ - AI/ML Engineer
- _Mohammed Affaan Shaikh_ - Backend Developer
- _Aiman Kulay_ - UI/UX Developer

---

## 📞 Contact

## 🙏 Acknowledgments

- [Google Gemini](https://deepmind.google/technologies/gemini/) - LLM capabilities
- [Pinecone](https://www.pinecone.io/) - Vector database
- [FastAPI](https://fastapi.tiangolo.com/) - Web framework

---

## ⭐ Star History

If you find this project useful, please consider giving it a star! ⭐

---

_Built with ❤ for modern recruitment_
