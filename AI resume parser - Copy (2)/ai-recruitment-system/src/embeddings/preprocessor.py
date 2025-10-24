"""
Preprocessor: Clean and prepare resume JSON for embedding generation
"""
from typing import Dict, List, Any, Optional
import json

class ResumePreprocessor:
    """Clean and format resume JSON Data for embedding generation"""

    def __init__(self, max_length: int = 8000):
        """
        Initialize preprocessor
        
        Args:
            max_length: Maximum text length for embedding (default: 8000 chars)
        """
        self.max_length = max_length

    def process(self, resume_json: Dict[str, Any]) -> str:
        """
        Convert resume JSON to clean text for embedding
        
        Args:
            resume_json: Parsed resume data as dictionary
            
        Returns:
            Cleaned text ready for embedding generation
        """
        text_parts = []
        
        # 1. Professional Summary
        if resume_json.get('professional_summary'):
            text_parts.append(f"Summary: {resume_json['professional_summary']}")
        
        # 2. Skills (most important for matching)
        skills = resume_json.get('skills', [])
        if skills:
            # Handle both list of strings and list of objects
            skill_names = self._extract_skill_names(skills)
            if skill_names:
                text_parts.append(f"Skills: {', '.join(skill_names)}")
        
        # 3. Work Experience
        work_exp = resume_json.get('work_experience', [])
        if work_exp:
            exp_text = self._format_work_experience(work_exp)
            if exp_text:
                text_parts.append(f"Experience: {exp_text}")

        # 4. Education
        education = resume_json.get('education', [])
        if education:
            edu_text = self._format_education(education)
            if edu_text:
                text_parts.append(f"Education: {edu_text}")

        # 5. Certifications
        certifications = resume_json.get('certifications', [])
        if certifications:
            cert_text = self._format_certifications(certifications)
            if cert_text:
                text_parts.append(f"Certifications: {cert_text}")

        # 6 Projects 
        projects = resume_json.get('projects', [])
        if projects:
            project_text = self._format_projects(projects)
            if project_text:
                text_parts.append(f"Projects: {project_text}")

        
        #Combine all parts
        full_text = " | ".join(text_parts)

        #Truncate if too long
        if len(full_text) > self.max_length:
            full_text = full_text[:self.max_length] + "..."
        
        return full_text
    
    def _extract_skill_names(self, skills: List[Any]) -> List[str]:
        """Extract skill names from various formats"""
        skill_names = []

        for skill in skills:
            if isinstance(skill, str):
                skill_names.append(skill)
            elif isinstance(skill, dict) and 'name' in skill:
                skill_names.append(skill['name'])
            elif hasattr(skill, 'name'):
                skill_names.append(skill.name)

        return [s for s in skill_names if s and s.strip()]
    
    def _format_work_experience(self, work_exp: List[Dict]) -> str:
        """Format ALL work experience into text"""
        exp_parts = []

        for exp in work_exp:  # ✅ REMOVED [:5] - Include ALL jobs
            parts = []

            # Job title and company
            if exp.get('job_title'):
                parts.append(exp['job_title'])
            if exp.get('company_name'):
                parts.append(f"at {exp['company_name']}")

            # Duration
            start = exp.get('start_date')
            end = exp.get('end_date') or 'Present' if exp.get('is_current') else exp.get('end_date')
            if start:
                duration = f"({start} to {end})" if end else f"(since {start})"
                parts.append(duration)

            # Description (FULL description, not truncated)
            if exp.get('description'):
                parts.append(f"- {exp['description']}")  # ✅ No [:200] limit

            # Key responsibilities (ALL of them)
            responsibilities = exp.get('key_responsibilities', [])
            if responsibilities:
                resp_text = '; '.join(responsibilities)  # ✅ No [:3] limit
                parts.append(f"Responsibilities: {resp_text}")

            if parts:
                exp_parts.append(' '.join(parts))

        return ' | '.join(exp_parts)

    def _format_education(self, education: List[Dict]) -> str:
        """Format ALL education into text"""
        edu_parts = []
        
        for edu in education:  # ✅ REMOVED [:3] - Include ALL education
            parts = []
            
            if edu.get('degree'):
                parts.append(edu['degree'])
            if edu.get('field_of_study'):
                parts.append(f"in {edu['field_of_study']}")
            if edu.get('institution'):
                parts.append(f"from {edu['institution']}")
            if edu.get('end_date'):
                parts.append(f"({edu['end_date']})")
            if edu.get('gpa'):
                parts.append(f"GPA: {edu['gpa']}")  # ✅ Added GPA
            
            if parts:
                edu_parts.append(' '.join(parts))
        
        return ', '.join(edu_parts)
    
    def _format_certifications(self, certifications: List[Dict]) -> str:
        """Format ALL certifications into text"""
        cert_names = []
        
        for cert in certifications:  # ✅ REMOVED [:10] - Include ALL certs
            if cert.get('name'):
                cert_names.append(cert['name'])
        
        return ', '.join(cert_names)
    
    def _format_projects(self, projects: List[Dict]) -> str:
        """Format ALL projects into text"""
        project_parts = []
        
        for project in projects:  # ✅ REMOVED [:5] - Include ALL projects
            if project.get('name'):
                parts = [project['name']]
                if project.get('description'):
                    parts.append(project['description'])  # ✅ Full description
                project_parts.append(' - '.join(parts))
        
        return ' | '.join(project_parts)
    
    def validate_input(self, resume_json: Dict[str, Any]) -> bool:
        """
        Validate that resume JSON has minimum required fields
        
        Args:
            resume_json: Resume data dictionary
            
        Returns:
            True if valid, False otherwise
        """
        #check required fields
        required_fields = ['full_name', 'email', 'skills']

        for field in required_fields:
            if not resume_json.get(field):
                print(f"❌ Missing required field: {field}")
                return False
        
        # Check that skills is not empty
        skills = resume_json.get('skills', [])
        if not skills or len(skills) == 0:
            print("❌ Skills list is empty")
            return False
        
        return True 
    
    def process_large_resume(self, resume_json: Dict) -> List[str]:
        """Split large resumes into 3 semantic chunks"""  

        chunks = [] 

        # Chunk 1: Skills + Summary (~500 tokens)
        chunk1 = []
        if resume_json.get('professional_summary'):
            chunk1.append(resume_json['professional_summary'][:8000])
        chunk1.append(f"Skills: {', '.join(resume_json.get('skills', []))}")
        chunks.append(" | ".join(chunk1))

        #Chunk2 : work experience (~ 1500 tokens)
        exp_text = self._format_work_experience(resume_json.get('work_experience', []))
        chunks.append(f"Experience: {exp_text[:6000]}")

        # Chunk 3: Education + Projects + Certs (~500 tokens)
        chunk3 = []
        chunk3.append(self._format_education(resume_json.get('education', [])))
        if resume_json.get('projects'):
            chunk3.append(self._format_projects(resume_json['projects']))
        if resume_json.get('certifications'):
            chunk3.append(self._format_certifications(resume_json['certifications']))
        chunks.append(" | ".join(chunk3))

        return chunks
    


    # Utility function for quick processing
def preprocess_resume(resume_json: Dict[str, any]) -> str:
    """
    Quick utility function to preprocess a resume
    Args:
        resume_json: Resume data as dictionary
    Returns:
        Preprocessed text ready for embedding
    """
    preprocessor = ResumePreprocessor()
    return preprocessor.process(resume_json)

if __name__ == "__main__":
    # Test the preprocessor with sample data
    sample_resume = {
        "full_name": "John Doe",
        "email": "john@example.com",
        "professional_summary": "Experienced software engineer with 5 years in Python and ML",
        "skills": ["Python", "Machine Learning", "FastAPI", "Docker"],
        "work_experience": [
            {
                "job_title": "Senior Software Engineer",
                "company_name": "Tech Corp",
                "start_date": "2020-01-01",
                "end_date": None,
                "is_current": True,
                "description": "Leading ML infrastructure team",
                "key_responsibilities": ["Design ML pipelines", "Mentor junior engineers"]
            }
        ],
        "education": [
            {
                "degree": "Bachelor of Science",
                "field_of_study": "Computer Science",
                "institution": "University XYZ",
                "end_date": "2018-05-01"
            }
        ]
    }
    preprocessor = ResumePreprocessor()
    # Validate
    if preprocessor.validate_input(sample_resume):
        print("✅ Validation passed")
        # Process
        processed_text = preprocessor.process(sample_resume)
        print(f"\n📝 Preprocessed Text ({len(processed_text)} chars):")
        print("-" * 80)
        print(processed_text)
        print("-" * 80)
    else:
        print("❌ Validation failed")