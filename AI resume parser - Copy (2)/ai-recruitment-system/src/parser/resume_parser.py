# # import os 
# # import json
# # import time
# # import pandas as pd
# # import fitz
# # from dotenv import load_dotenv
# # from datetime import date 
# # from enum import Enum
# # from PyPDF2 import PdfReader
# # from pydantic import BaseModel, Field, HttpUrl, EmailStr, field_validator
# # from typing import List, Optional, Dict, Any
# # from datetime import date as DateType

# # from google import genai
# # from google.genai import types

# # from langchain_google_genai import ChatGoogleGenerativeAI
# # from langchain_openai import ChatOpenAI

# # from langchain_core.prompts import PromptTemplate
# # from langchain_core.output_parsers import PydanticOutputParser
# # from langchain_core.messages import HumanMessage

# # class EmploymentType(str, Enum):
# #     FULL_TIME = "full_time"
# #     PART_TIME = "part_time"
# #     CONTRACT = "contract"
# #     INTERNSHIP = "internship"
# #     FREELANCE = "freelance"
# #     VOLUNTEER = "volunteer"


# # class EducationLevel(str, Enum):
# #     HIGH_SCHOOL = "high_school"
# #     DIPLOMA = "diploma"
# #     BACHELOR = "bachelor"
# #     MASTER = "master"
# #     DOCTORATE = "doctorate"
# #     CERTIFICATE = "certificate"



# # class WorkExperience(BaseModel):
# #     job_title: str = Field(description="Job title/position")
# #     company_name: str = Field(description="Company name")
# #     location: Optional[str] = Field(default=None, description="Work location")
# #     employment_type: Optional[EmploymentType] = Field(default=None, description="Type of employment")
# #     start_date: Optional[date] = Field(default=None, description="Start date")
# #     end_date: Optional[date] = Field(default=None, description="End date (None if current)")
# #     is_current: bool = Field(default=False, description="Is this current position")
# #     description: Optional[str] = Field(default=None, description="Job description")
# #     key_responsibilities: Optional[List[str]] = Field(default=None, description="Key responsibilities")


# # class Education(BaseModel):
# #     degree: str = Field(description="Degree name")
# #     institution: str = Field(description="School/university name")
# #     field_of_study: Optional[str] = Field(default=None, description="Major/field of study")
# #     education_level: Optional[EducationLevel] = Field(default=None, description="Level of education")
# #     start_date: Optional[date] = Field(default=None, description="Start date")
# #     end_date: Optional[date] = Field(default=None, description="Graduation date")
# #     gpa: Optional[float] = Field(default=None, description="GPA score")
# #     honors: Optional[List[str]] = Field(default=None, description="Honors, Dean's List, academic distinctions")

# #     @field_validator('gpa', mode='before')
# #     @classmethod
# #     def parse_gpa(cls, v):
# #         """Convert GPA strings like '8.65/10' or '84%' to float"""
# #         if v is None:
# #             return None
# #         if isinstance(v, (int, float)):
# #             return float(v)
# #         if isinstance(v, str):
# #             v = v.strip().replace('%', '')
# #             if '/' in v:
# #                 try:
# #                     return float(v.split('/')[0].strip())
# #                 except:
# #                     return None
# #             try:
# #                 return float(v)
# #             except:
# #                 return None
# #         return None

# # class Award(BaseModel):
# #     """Award or achievement"""
# #     title: str
# #     issuer: Optional[str] = None
# #     date: Optional[DateType] = None
# #     description: Optional[str] = None


# # class Project(BaseModel):
# #     name: str = Field(description="Project name")
# #     description: str = Field(description="Project description")
# #     github_url: Optional[HttpUrl] = Field(default=None, description="GitHub repository URL")
# #     project_url: Optional[HttpUrl] = Field(default=None, description="Live project URL or demo link")


# # class Certification(BaseModel):
# #     name: str = Field(description="Certification name")
# #     issuing_organization: Optional[str] = Field(description="Organization that issued the certification")
# #     issue_date: Optional[date] = Field(default=None, description="Date issued")
# #     expiry_date: Optional[date] = Field(default=None, description="Expiration date")
# #     credential_id: Optional[str] = Field(default=None, description="Credential or license number")
# #     credential_url: Optional[HttpUrl] = Field(default=None, description="Verification URL")


# # class Publication(BaseModel):
# #     title: str = Field(description="Publication title")
# #     description: Optional[str] = Field(default=None, description="Brief description or abstract")
# #     url: Optional[HttpUrl] = Field(default=None, description="Publication URL")
# #     publication_date: Optional[date] = Field(default=None, description="Publication date")

# # class VolunteerExperience(BaseModel):
# #     """Volunteer work and community service"""
# #     role: str = Field(description="Volunteer role or position")
# #     organization: str = Field(description="Organization or cause name")
# #     location: Optional[str] = Field(default=None, description="Location of volunteer work")
# #     start_date: Optional[date] = Field(default=None, description="Start date")
# #     end_date: Optional[date] = Field(default=None, description="End date (None if ongoing)")
# #     is_current: bool = Field(default=False, description="Is this ongoing volunteer work")
# #     description: Optional[str] = Field(default=None, description="Description of volunteer work and impact")


# # class ResumeData(BaseModel):
# #     """Streamlined industrial-grade resume data model"""
    
# #     # Basic Information (Required)
# #     full_name: str = Field(description="Full name of the candidate")
# #     email: EmailStr = Field(description="Email address")
# #     phone_number: Optional[str] = Field(default=None, description="Phone number")
# #     location: Optional[str] = Field(default=None, description="Current location/address")
    
# #     # Professional Identity
# #     professional_title: Optional[str] = Field(default=None, description="Current job title or desired position")
# #     professional_summary: Optional[str] = Field(default=None, description="Professional summary/objective")
    
# #     # Online Presence
# #     linkedin_url: Optional[HttpUrl] = Field(default=None, description="LinkedIn profile URL")
# #     github_url: Optional[HttpUrl] = Field(default=None, description="GitHub profile URL")
# #     portfolio_url: Optional[HttpUrl] = Field(default=None, description="Portfolio website URL")
# #     personal_website: Optional[HttpUrl] = Field(default=None, description="Personal website or blog URL")

# #     # Core Sections (Required)
# #     skills: List[str] = Field(description="List of skills")
# #     work_experience: List[WorkExperience] = Field(description="Work experience history")
# #     education: List[Education] = Field(description="Education history")
    
# #     # Optional Sections
# #     projects: Optional[List[Project]] = Field(default=None, description="Projects")
# #     publications: Optional[List[Publication]] = Field(default=None, description="Publications and research papers")
# #     certifications: Optional[List[Certification]] = Field(default=None, description="Certifications")
# #     volunteer_experience: Optional[List[VolunteerExperience]] = Field(default=None, description="Volunteer work and community service")
# #     languages: Optional[List[str]] = Field(default=None, description="Spoken languages (NOT programming)")
# #     achievements: Optional[List[str]] = Field(default=None, description="Awards, recognitions, and notable achievements")
# #     awards: Optional[List[Award]] = Field(default_factory=list)
# #     personal_interests: Optional[List[str]] = Field(default=None, description="Hobbies, interests, and extracurricular activities")


# # class ResumeParser:
# #     """AI-powered resume parser using LangChain and LLM"""
    
# #     def __init__(self, api_key: str = None):
# #         """Initialize the resume parser"""
# #         load_dotenv()
# #         # self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
# #         self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
# #         if not self.api_key:
# #             # raise ValueError("Google API key not found. Set GOOGLE_API_KEY environment variable.")
# #             raise ValueError("OpenRouter API key not found. Set OPENROUTER_API_KEY environment variable.")
        
# #         self.parser = PydanticOutputParser(pydantic_object=ResumeData)
        
#         # # CRITICAL FIX: Increase max_output_tokens
#         # self.llm = ChatGoogleGenerativeAI(
#         #     temperature=0, 
#         #     model="gemini-2.5-flash",
#         #     # model = 'gemini-2.0-flash-001',
#         #     google_api_key=self.api_key,
#         #     max_output_tokens=8192,
#         #     streaming = True
#         # )
# #         self.llm = ChatOpenAI(
# #            model="openai/gpt-4o-mini",
# #            temperature=0,
# #            api_key=self.api_key,
# #            base_url="https://openrouter.ai/api/v1",
# #            max_tokens=8192,
# #            default_headers={
# #                "HTTP-Referer": "https://github.com/shahnawaz-4iamericas/ai-recruitment-engine",
# #                "X-Title": "AI Resume Parser"
# #            }
# #         )
        
# #         # Create the prompt template
# #         self.template = PromptTemplate(
# #             template="""You are an expert resume parser. Extract ALL information and output COMPLETE valid JSON.

# # CRITICAL: You MUST return the FULL JSON response. Do NOT truncate. Generate complete JSON for all sections.

# # Extract these fields from the resume:


# # REQUIRED FIELDS (Must always be present):


# # 1. full_name: "Full Name"
# # 2. email: "email@example.com"
# # 3. skills: ["Skill1", "Skill2", "Skill3"] - ARRAY OF STRINGS ONLY.

# # 4. work_experience: [
# #    {{
# #      "job_title": "Position Title",
# #      "company_name": "Company Name",
# #      "location": "City, State" or "Remote (City, State)" or null,
# #      "employment_type": "full_time|part_time|contract|freelance|internship|volunteer",
# #      "start_date": "2020-01-01" or null,
# #      "end_date": "2022-12-31" or null,
# #      "is_current": false,
# #      "description": "Brief overview",
# #      "key_responsibilities": ["Responsibility 1", "Responsibility 2"]
# #    }}
# #    - Extract location even if it says "Remote", "San Francisco, CA", etc.
# #    - For multiple roles at same company, create separate entries
   
# # 5. education: [
# #    {{
# #      "degree": "Bachelor of Science" or "BS" or "BSc",
# #      "institution": "University Name",
# #      "field_of_study": "Computer Science, Minor: Mathematics",
# #      "education_level": "bachelor|master|doctorate|diploma|certificate",
# #      "start_date": null or "2012-09-01",
# #      "end_date": "2016-05-01",
# #      "gpa": 3.8,
# #      "honors": ["Dean's List Fall 2015", "Summa Cum Laude", "Magna Cum Laude"],
# #      "relevant_coursework": ["Only if explicitly listed in resume"]
# #    }}
# # ]
# #    - Extract GPA from formats like "3.8/4.0", "GPA: 3.8", "8.5/10"

# # OPTIONAL FIELDS (if present in resume):

# # - phone_number: "+1 (555) 123-4567" or "555-123-4567" or "(555) 123-4567"
# # - location: "City, State ZIP" or "City, State, Country"
# # - professional_title: "Senior Software Engineer | Full-Stack Developer"
# # - professional_summary: "Brief professional summary paragraph"

# # ONLINE PRESENCE (Add https:// if missing):
# # - linkedin_url, github_url, portfolio_url, personal_website (add https:// if missing)

# # PROJECTS:
# # - projects: [
# #    {{
# #      "name": "Project Name",
# #      "description": "Description",
# #      "github_url": "https://github.com/user/repo" or null,
# #      "project_url": "https://liveproject.com" or null
# #    }}
# # ]

# # PUBLICATIONS & SPEAKING:
# # - publications: [
# #    {{
# #      "title": "Title",
# #      "description": "Conference name, role",
# #      "url": "https://publication-url.com" or null,
# #      "publication_date": "2024-10-01" or null
# #    }}
# # ]
# # CERTIFICATIONS & LICENSES:
# # - certifications: [
# #    {{
# #      "name": "Name",
# #      "issuing_organization": "Organization Name",
# #      "issue_date": "2023-01-01" or null,
# #      "expiry_date": "2026-01-01" or null,
# #      "credential_id": "RN-278945" or null,
# #      "credential_url": "https://verify-url.com" or null
# #    }}
# # ]
# # VOLUNTEER EXPERIENCE:
# # - volunteer_experience: [
# #    {{
# #      "role": "Volunteer Role",
# #      "organization": "Organization Name",
# #      "location": "City, State" or null,
# #      "start_date": "2020-01-01" or null,
# #      "end_date": null,
# #      "is_current": true,
# #      "description": "Description of volunteer work and impact"
# #    }}
# # ]

# # AWARDS & ACHIEVEMENTS:
# # - awards: [
# #    {{
# #      "title": "Award Name or Achievement",
# #      "issuer": "Organization",
# #      "date": "2024-01-01" or null,
# #      "description": "Context or details about the award"
# #    }}
# # ]

# # - languages: ["English (Native)"] - Human languages only, NOT programming


# # - personal_interests: ["Photography", "Blogging", "Hiking", "Chess"]
# #    - Look for sections: "Interests", "Hobbies", "Personal", "Outside of Work"
# # iate"
# # - "bachelor"
# # - "master"
# # - "doctorate"
# # - "certificate"

# # DATE PARSING RULES:


# # Convert ALL dates to YYYY-MM-DD format:
# # - "March 2021" → "2021-03-01"
# # - "Jan 2019" → "2019-01-01"
# # - "08/2016" → "2016-08-01"
# # - "2020" → "2020-01-01"
# # - "Present/Current" → null (set is_current: true)

# # CRITICAL RULES:
# # ✓ Use null for missing values, [] for empty arrays
# # ✓ URLs must have http:// or https://
# # ✓ Programming languages → skills (NOT languages)
# # ✓ Extract locations from work experience
# # ✓ NO hallucinated data - only extract what exists
# # ✓ NEVER invent relevant_coursework - extract only if explicitly listed
# # ✓ Remove escape characters (\\, \\")


# # RESUME TEXT:

# # {resume_text}


# # EXTRACTED LINKS:

# # {extracted_links}


# # OUTPUT INSTRUCTIONS:

# # Return COMPLETE valid JSON matching the ResumeData schema.
# # DO NOT truncate any section. Include ALL information found.
# # Ensure proper JSON formatting with correct quotes and commas.

# # START JSON OUTPUT NOW:""",
# #             input_variables=["resume_text", "extracted_links"]
# import os 
# import uuid

# import json
# import time
# import pandas as pd
# import fitz
# from dotenv import load_dotenv
# from datetime import date 
# from enum import Enum
# from PyPDF2 import PdfReader
# from pydantic import BaseModel, Field, HttpUrl, EmailStr, field_validator
# from typing import List, Optional, Dict, Any
# from datetime import date as DateType
# from datetime import datetime

# from google import genai
# from google.genai import types

# from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_openai import ChatOpenAI

# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import PydanticOutputParser
# from langchain_core.messages import HumanMessage

# class EmploymentType(str, Enum):
#     FULL_TIME = "full_time"
#     PART_TIME = "part_time"
#     CONTRACT = "contract"
#     INTERNSHIP = "internship"
#     FREELANCE = "freelance"


# class EducationLevel(str, Enum):
#     HIGH_SCHOOL = "high_school"
#     DIPLOMA = "diploma"
#     BACHELOR = "bachelor"
#     MASTER = "master"
#     DOCTORATE = "doctorate"
#     CERTIFICATE = "certificate"


# class WorkExperience(BaseModel):
#     job_title: str = Field(description="Job title/position")
#     company_name: str = Field(description="Company name")
#     location: Optional[str] = Field(default=None, description="Work location")
#     employment_type: Optional[EmploymentType] = Field(default=None, description="Type of employment")
#     start_date: Optional[date] = Field(default=None, description="Start date")
#     end_date: Optional[date] = Field(default=None, description="End date (None if current)")
#     is_current: bool = Field(default=False, description="Is this current position")
#     description: Optional[str] = Field(default=None, description="Job description")
#     key_responsibilities: Optional[List[str]] = Field(default=None, description="Key responsibilities")

# class Education(BaseModel):
#     degree: str = Field(description="Degree name")
#     institution: str = Field(description="School/university name")
#     field_of_study: Optional[str] = Field(default=None, description="Major/field of study")
#     education_level: Optional[EducationLevel] = Field(default=None, description="Level of education")
#     start_date: Optional[date] = Field(default=None, description="Start date")
#     end_date: Optional[date] = Field(default=None, description="Graduation date")
#     gpa: Optional[float] = Field(default=None, description="GPA score")

#     @field_validator('gpa', mode='before')
#     @classmethod
#     def parse_gpa(cls, v):
#         """Convert GPA strings like '8.65/10' or '84%' to float"""
#         if v is None:
#             return None
#         if isinstance(v, (int, float)):
#             return float(v)
#         if isinstance(v, str):
#             v = v.strip().replace('%', '')
#             if '/' in v:
#                 try:
#                     return float(v.split('/')[0].strip())
#                 except:
#                     return None
#             try:
#                 return float(v)
#             except:
#                 return None
#         return None


# class Project(BaseModel):
#     name: str = Field(alias="project_name", description="Project name")
#     description: str = Field(description="Project description")
#     github_url: Optional[HttpUrl] = Field(default=None, description="GitHub repository URL")
#     project_url: Optional[HttpUrl] = Field(default=None, description="Live project URL or demo link")

#     class Config:
#         populate_by_name = True 

# class Certification(BaseModel):
#     name: str = Field(description="Certification name")
#     issuing_organization: Optional[str] = Field(description="Organization that issued the certification")
#     issue_date: Optional[date] = Field(default=None, description="Date issued")
#     credential_url: Optional[HttpUrl] = Field(default=None, description="Verification URL")


# class ResumeData(BaseModel):
#     """Streamlined resume data model for AI recruitment"""
    
#     # Basic Information (Required)
#     # full_name: Optional[str] = Field(default="Unknown Candidate", description="Full name")
#     # email: Optional[EmailStr] = Field(default=None, description="Email address"
#     full_name: str = Field(default="Anonymous Candidate")
#     email: str = Field(default="anonymous@system.generated")
#     phone_number: Optional[str] = None

#     # phone_number: Optional[str] = Field(default=None, description="Phone number")
#     location: Optional[str] = Field(default=None, description="Current location/address")
    
#     # Professional Identity
#     professional_title: Optional[str] = Field(default=None, description="Current job title or desired position")
#     professional_summary: Optional[str] = Field(default=None, description="Professional summary/objective")
    
#     # Online Presence
#     linkedin_url: Optional[HttpUrl] = Field(default=None, description="LinkedIn profile URL")
#     github_url: Optional[HttpUrl] = Field(default=None, description="GitHub profile URL")
#     portfolio_url: Optional[HttpUrl] = Field(default=None, description="Portfolio website URL")

#     # Core Sections (Required)
#     skills: List[str] = Field(description="List of skills")
#     work_experience: List[WorkExperience] = Field(description="Work experience history")
#     education: List[Education] = Field(description="Education history")
    
#     # Optional Sections
#     projects: Optional[List[Project]] = Field(default=None, description="Projects")
#     certifications: Optional[List[Certification]] = Field(default=None, description="Certifications")
#     languages: Optional[List[str]] = Field(default=None, description="Spoken languages (NOT programming)")
#     achievements: Optional[List[str]] = Field(default=None, description="Awards and notable achievements")


# class ResumeParser:
#     """AI-powered resume parser using LangChain and LLM"""
    
#     def __init__(self, api_key: str = None):
#         """Initialize the resume parser"""
#         load_dotenv()
#         # self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
#         self.api_key = api_key or os.getenv("GOOGLE_API_KEY")

#         if not self.api_key:
#             raise ValueError("OpenRouter API key not found. Set OPENROUTER_API_KEY environment variable.")
        
#         self.parser = PydanticOutputParser(pydantic_object=ResumeData)
        
#         # Option 1: Google Gemini (Fastest - Recommended)
#         self.llm = ChatGoogleGenerativeAI(
#             temperature=0, 
#             model='gemini-2.0-flash-exp',  # Fastest free model
#             google_api_key=os.getenv("GOOGLE_API_KEY"),  # Separate API key needed
#             max_output_tokens=4096,
#             streaming=True
#         )
        
#         # Option 2: Gemini via OpenRouter (if you prefer one API key)
#         # self.llm = ChatOpenAI(
#         #     model="google/gemini-2.0-flash-exp:free",
#         #     temperature=0,
#         #     api_key=self.api_key,
#         #     base_url="https://openrouter.ai/api/v1",
#         #     max_tokens=4096,
#         #     streaming=True,
#         #     default_headers={
#         #         "HTTP-Referer": "https://github.com/shahnawaz-4iamericas/ai-recruitment-engine",
#         #         "X-Title": "AI Resume Parser"
#         #     }
#         # )
        
#         # Option 3: OpenAI GPT-4o-mini (Original - slower but highest quality)
#         # self.llm = ChatOpenAI(
#         #     model="openai/gpt-4o-mini",
#         #     temperature=0,
#         #     api_key=self.api_key,
#         #     base_url="https://openrouter.ai/api/v1",
#         #     max_tokens=4096,
#         #     streaming=True,
#         #     default_headers={
#         #         "HTTP-Referer": "https://github.com/shahnawaz-4iamericas/ai-recruitment-engine",
#         #         "X-Title": "AI Resume Parser"
#         #     }
#         # )
        
#         # Optimized minimal prompt
#         self.template = PromptTemplate(
#             template="""Extract resume data and return valid JSON. Use null for missing values, [] for empty arrays.

# REQUIRED FIELDS:
# - full_name, email, skills (array), work_experience (array), education (array)

# OPTIONAL FIELDS:
# - phone_number, location, professional_title, professional_summary
# - linkedin_url, github_url, portfolio_url
# - projects, certifications, languages (human languages only), achievements

# WORK EXPERIENCE FORMAT:
# {{
#   "job_title": "string",
#   "company_name": "string", 
#   "location": "string or null",
#   "employment_type": "full_time|part_time|contract|internship|freelance",
#   "start_date": "YYYY-MM-DD or null",
#   "end_date": "YYYY-MM-DD or null",
#   "is_current": boolean,
#   "description": "string or null",
#   "key_responsibilities": ["array"] or null
# }}

# EDUCATION FORMAT:
# {{
#   "degree": "string",
#   "institution": "string",
#   "field_of_study": "string or null",
#   "education_level": "bachelor|master|doctorate|diploma|certificate",
#   "start_date": "YYYY-MM-DD or null",
#   "end_date": "YYYY-MM-DD or null",
#   "gpa": number or null
# }}

# PROJECTS FORMAT
# {{
#   "name": "Project Name",
#   "description": "Description",
#   "github_url": "https://github.com/user/repo" or null,
#   "project_url": "https://liveproject.com" or null
# }}
# ]

# CERTIFICATIONS & LICENSES:
# {{
#   "name": "Name",
#   "issuing_organization": "Organization Name",
#   "issue_date": "2023-01-01" or null,
#   "expiry_date": "2026-01-01" or null,
#   "credential_id": "RN-278945" or null,
#   "credential_url": "https://verify-url.com" or null
# }}

# DATE CONVERSION:
# "March 2021" → "2021-03-01", "2020" → "2020-01-01", "Present" → null (set is_current: true)

# RULES:
# - URLs must include http:// or https://
# - Programming languages go in skills, NOT languages
# - GPA: extract from "3.8/4.0" → 3.8
# - NO invented data

# RESUME TEXT:
# {resume_text}

# EXTRACTED LINKS:
# {extracted_links}

# Return complete valid JSON:""",
#             input_variables=["resume_text", "extracted_links"]
#         )
#         # Create chain WITHOUT the parser - we'll manually parse and validate
#         self.chain = self.template | self.llm

#     def extract_text_and_links_from_pdf(self, pdf_path: str) -> tuple:
#         """Extract text and hyperlinks from PDF using PyMuPDF"""
#         try:
#             doc = fitz.open(pdf_path)
#             text = ""
#             links = {}
            
#             for page_num in range(len(doc)):
#                 page = doc[page_num]
#                 text += page.get_text()
                
#                 # Extract links
#                 page_links = page.get_links()
#                 for link in page_links:
#                     if 'uri' in link:
#                         rect = fitz.Rect(link['from'])
#                         link_text = page.get_textbox(rect).strip()
#                         links[link_text.lower()] = link['uri']
            
#             doc.close()
#             return text, links
            
#         except Exception as e:
#             raise Exception(f"Error extracting text from PDF: {str(e)}")


#     # def parse_resume(self, pdf_path: str) -> dict:
#     #     """Parse a single resume PDF and return structured data"""

#     #     # Start total timer
#     #     start_time = time.time()

#     #     try:
#     #         # Time PDF extraction
#     #         pdf_start = time.time()
#     #         resume_text, extracted_links = self.extract_text_and_links_from_pdf(pdf_path)
#     #         pdf_time = time.time() - pdf_start

#     #         # Time LLM API call
#     #         llm_start = time.time()
#     #         response = self.chain.invoke({
#     #             "resume_text": resume_text, 
#     #             "extracted_links": str(extracted_links)
#     #         })
#     #         llm_time = time.time() - llm_start

#     #         # Time JSON parsing and validation
#     #         parse_start = time.time()
#     #         json_str = response.content.strip()
#     #         if "```json" in json_str:
#     #             json_str = json_str.split("```json")[1].split("```")[0].strip()

#     #         data = json.loads(json_str)
#     #         data = self._clean_resume_data(data)
#     #         validated = ResumeData(**data)
#     #         self._last_parsed_object = validated

#     #         result = validated.model_dump()
#     #         result = self._serialize_httpurl(result)
#     #         parse_time = time.time() - parse_start

#     #         # Calculate total time
#     #         total_time = time.time() - start_time

#     #         # Print timing breakdown
#     #         print(f"\n{'='*50}")
#     #         print(f"Parsing completed for: {pdf_path}")
#     #         print(f"{'='*50}")
#     #         print(f"PDF Extraction:    {pdf_time:.3f}s ({pdf_time/total_time*100:.1f}%)")
#     #         print(f"LLM API Call:      {llm_time:.3f}s ({llm_time/total_time*100:.1f}%)")
#     #         print(f"JSON Processing:   {parse_time:.3f}s ({parse_time/total_time*100:.1f}%)")
#     #         print(f"TOTAL TIME:        {total_time:.3f}s")
#     #         print(f"{'='*50}\n")

#     #         return result
        
#     #     except Exception as e:
#     #         total_time = time.time() - start_time
#     #         print(f"Error after {total_time:.3f}s: {str(e)}")
#     #         raise Exception(f"Error parsing resume: {str(e)}")
#     def parse_resume(self, pdf_path: str) -> dict:
#         """Parse a single resume PDF and return structured data"""
#         start_time = time.time()

#         try:
#             # Time PDF extraction
#             pdf_start = time.time()
#             resume_text, extracted_links = self.extract_text_and_links_from_pdf(pdf_path)
#             pdf_time = time.time() - pdf_start

#             # Truncate if too long
#             if len(resume_text) > 20000:  # ~15K tokens
#                 print(f"   ⚠️ Very long resume, truncating to 20K chars")
#                 resume_text = resume_text[:20000]

#             # Time LLM API call
#             llm_start = time.time()
#             response = self.chain.invoke({
#                 "resume_text": resume_text, 
#                 "extracted_links": str(extracted_links)
#             })
#             llm_time = time.time() - llm_start

#             # Time JSON parsing and validation
#             parse_start = time.time()
#             json_str = response.content.strip()
#             if "```json" in json_str:
#                 json_str = json_str.split("```json")[1].split("```")[0].strip()

#             # 🆕 TRY to parse JSON, fallback if fails
#             try:
#                 data = json.loads(json_str)
#             except json.JSONDecodeError as e:
#                 print(f"   ⚠️ JSON parsing failed, using fallback data")
#                 data = {
#                 "full_name": "Parse Failed",
#                 "email": "parsefailed@system.com",
#                 "professional_summary": "Resume parsing failed due to malformed data",  # 🆕 Add this
#                 "skills": ["Data extraction failed"],  # 🆕 Add dummy skill
#                 "work_experience": [],
#                 "education": []
#             }

#             # 🆕 Clean null values BEFORE validation
#             if not data.get('full_name'):
#                 data['full_name'] = "Anonymous Candidate"
#             if not data.get('email'):
#                 data['email'] = "anonymous@system.com"
#             if not data.get('professional_title'):  # 🆕 ADD THIS
#                 data['professional_title'] = "" 

#             # 🆕 Clean work experience nulls
#             if data.get('work_experience'):
#                 for exp in data['work_experience']:
#                     if not exp.get('company_name'):
#                         exp['company_name'] = "Unknown Company"
#                     if not exp.get('job_title'):
#                         exp['job_title'] = "Unknown Position"


#             # 🆕 Fix education nulls
#             if data.get('education'):
#                 for edu in data['education']:
#                     if not edu.get('institution'):
#                         edu['institution'] = "Unknown"
#                     if not edu.get('degree'):  # 🆕 Add this
#                         edu['degree'] = "Unknown Degree"  # 🆕 Add thi

#             data = self._clean_resume_data(data)
#             validated = ResumeData(**data)
#             self._last_parsed_object = validated

#             result = validated.model_dump()
#             result = self._serialize_httpurl(result)
#             parse_time = time.time() - parse_start

#             # Print timing
#             total_time = time.time() - start_time
#             print(f"\n{'='*50}")
#             print(f"Parsing completed for: {pdf_path}")
#             print(f"{'='*50}")
#             print(f"PDF Extraction:    {pdf_time:.3f}s ({pdf_time/total_time*100:.1f}%)")
#             print(f"LLM API Call:      {llm_time:.3f}s ({llm_time/total_time*100:.1f}%)")
#             print(f"JSON Processing:   {parse_time:.3f}s ({parse_time/total_time*100:.1f}%)")
#             print(f"TOTAL TIME:        {total_time:.3f}s")
#             print(f"{'='*50}\n")

#             return result

#         except Exception as e:
#             total_time = time.time() - start_time
#             print(f"Error after {total_time:.3f}s: {str(e)}")
#             raise Exception(f"Error parsing resume: {str(e)}")

#     def _serialize_httpurl(self, data):
#         """Convert HttpUrl and date objects to strings for JSON serialization"""
#         if isinstance(data, dict):
#             return {k: self._serialize_httpurl(v) for k, v in data.items()}
#         elif isinstance(data, list):
#             return [self._serialize_httpurl(item) for item in data]
#         elif hasattr(data, '__class__') and data.__class__.__name__ == 'HttpUrl':
#             return str(data)
#         elif isinstance(data, date):
#             return data.isoformat()
#         else:
#             return data

#     def create_flat_dataframe(self, resume_data) -> pd.DataFrame:
#         """Convert structured ResumeData to analysis-ready DataFrame"""
#         # Handle both dict and ResumeData object
#         if isinstance(resume_data, dict):
#             # Try to get the stored Pydantic object first
#             if hasattr(self, '_last_parsed_object'):
#                 resume_data = self._last_parsed_object
#             else:
#                 # Convert dict to ResumeData object
#                 try:
#                     resume_data = ResumeData(**resume_data)
#                 except Exception as e:
#                     print(f"Warning: Could not convert dict to ResumeData: {e}")
#                     # Continue with dict if conversion fails
#                     return self._create_flat_dataframe_from_dict(resume_data)

#         # Safe join function to handle None values and complex objects
#         def safe_join(data, default=''):
#             if data is None:
#                 return default
#             if isinstance(data, list):
#                 if not data:  # Empty list
#                     return default
#                 # Handle list of objects with name attribute
#                 if hasattr(data[0], 'name'):
#                     return ', '.join(item.name for item in data)
#                 # Handle list of strings
#                 elif isinstance(data[0], str):
#                     return ', '.join(data)
#                 # Handle list of dicts
#                 elif isinstance(data[0], dict):
#                     return ', '.join(str(item.get('name', item)) for item in data)
#                 else:
#                     return ', '.join(str(item) for item in data)
#             if isinstance(data, str):
#                 return data
#             return str(data)
    
#     def _parse_with_json_mode(self, resume_text: str, extracted_links: dict) -> dict:
#         """Fallback parsing method using JSON mode"""
#         try:
#             # Create prompt without format_instructions
#             prompt = self.template.format(
#                 resume_text=resume_text,
#                 extracted_links=str(extracted_links)
#             )
            
#             # Add schema hint
#             prompt += f"\n\nJSON Schema to follow:\n{json.dumps(ResumeData.model_json_schema(), indent=2)}"
            
#             # Direct LLM call
#             response = self.llm.invoke([HumanMessage(content=prompt)])
            
#             # Parse JSON from response
#             json_str = response.content.strip()
#             if "```json" in json_str:
#                 json_str = json_str.split("```json")[1].split("```")[0].strip()
            
#             # Parse and validate with less strict validation
#             data = json.loads(json_str)
            
#             # Clean data before validation
#             data = self._clean_resume_data(data)
            
#             validated = ResumeData(**data)
#             return validated.model_dump()
            
#         except Exception as e:
#             raise Exception(f"Fallback parsing failed: {str(e)}")
    
#     def _clean_resume_data(self, data: dict) -> dict:
#         """Clean and prepare data for Pydantic validation"""
#         # Convert GPA strings to floats (backup for validator)
#         if 'education' in data and data['education']:
#             for edu in data['education']:
#                 if 'gpa' in edu and edu['gpa'] is not None:
#                     if isinstance(edu['gpa'], str):
#                         gpa_str = edu['gpa'].strip().replace('%', '')
#                         if '/' in gpa_str:
#                             try:
#                                 edu['gpa'] = float(gpa_str.split('/')[0].strip())
#                             except:
#                                 edu['gpa'] = None
#                         else:
#                             try:
#                                 edu['gpa'] = float(gpa_str)
#                             except:
#                                 edu['gpa'] = None
#                     elif not isinstance(edu['gpa'], (int, float)):
#                         edu['gpa'] = None
        
#         # Convert language objects to simple strings
#         if 'languages' in data and data['languages']:
#             cleaned_languages = []
#             for lang in data['languages']:
#                 if isinstance(lang, dict):
#                     # Extract just the name if it's an object
#                     cleaned_languages.append(lang.get('name', str(lang)))
#                 elif isinstance(lang, str):
#                     cleaned_languages.append(lang)
#             data['languages'] = cleaned_languages if cleaned_languages else None
        
#         return data

#     def create_flat_dataframe(self, resume_data: ResumeData) -> pd.DataFrame:
#         """Convert structured ResumeData to analysis-ready DataFrame"""

#         # Safe join function to handle None values and complex objects
#         def safe_join(data, default=''):
#             if data is None:
#                 return default
#             if isinstance(data, list):
#                 if not data:  # Empty list
#                     return default
#                 # Handle list of objects with name attribute
#                 if hasattr(data[0], 'name'):
#                     return ', '.join(item.name for item in data)
#                 # Handle list of strings
#                 elif isinstance(data[0], str):
#                     return ', '.join(data)
#                 # Handle list of dicts
#                 elif isinstance(data[0], dict):
#                     return ', '.join(str(item.get('name', item)) for item in data)
#                 else:
#                     return ', '.join(str(item) for item in data)
#             if isinstance(data, str):
#                 return data
#             return str(data)

#         # Extract work experience info
#         current_job = resume_data.work_experience[0] if resume_data.work_experience else None
#         total_experience_years = len(resume_data.work_experience)

#         # Calculate years of experience more accurately
#         experience_duration = 0
#         if resume_data.work_experience:
#             for exp in resume_data.work_experience:
#                 if exp.start_date and exp.end_date:
#                     duration = exp.end_date.year - exp.start_date.year
#                     experience_duration += duration
#                 elif exp.start_date and exp.is_current:
#                     from datetime import date
#                     duration = date.today().year - exp.start_date.year
#                     experience_duration += duration

#         # Extract education info
#         latest_education = resume_data.education[0] if resume_data.education else None

#         # Extract skills by category
#         technical_skills = [skill.name for skill in resume_data.skills if skill.category and 'technical' in skill.category.lower()]
#         programming_skills = [skill.name for skill in resume_data.skills if skill.category and 'programming' in skill.category.lower()]

#         # Extract certifications info
#         active_certifications = []
#         if resume_data.certifications:
#             from datetime import date
#             today = date.today()
#             for cert in resume_data.certifications:
#                 if not cert.expiry_date or cert.expiry_date > today:
#                     active_certifications.append(cert.name)

#         flat_data = {
#             # Basic Information
#             'Full Name': resume_data.full_name,
#             'Email': resume_data.email,
#             'Phone Number': resume_data.phone_number,
#             'Location': resume_data.location,
#             'Professional Title': resume_data.professional_title,
#             'Career Level': resume_data.career_level,

#             # Online Presence
#             'LinkedIn': str(resume_data.linkedin_url) if resume_data.linkedin_url else '',
#             'GitHub': str(resume_data.github_url) if resume_data.github_url else '',
#             'Portfolio': str(resume_data.portfolio_url) if resume_data.portfolio_url else '',
#             'Personal Website': str(resume_data.personal_website) if resume_data.personal_website else '',

#             # Skills Analysis
#             'All Skills': safe_join([skill.name for skill in resume_data.skills]),
#             'Technical Skills': safe_join(technical_skills),
#             'Programming Skills': safe_join(programming_skills),
#             'Total Skills Count': len(resume_data.skills),
#             # 'Expert Level Skills': len([skill for skill in resume_data.skills if skill.level == SkillLevel.EXPERT]),

#             # Education
#             'Latest Degree': latest_education.degree if latest_education else '',
#             'Institution': latest_education.institution if latest_education else '',
#             'Field of Study': latest_education.field_of_study if latest_education else '',
#             'Education Level': latest_education.education_level.value if latest_education and latest_education.education_level else '',
#             'Graduation Year': latest_education.end_date.year if latest_education and latest_education.end_date else '',
#             'GPA': latest_education.gpa if latest_education else '',
#             'Academic Honors': latest_education.honors if latest_education else '',

#             # Work Experience
#             'Current Role': current_job.job_title if current_job else '',
#             'Current Company': current_job.company_name if current_job else '',
#             'Current Industry': current_job.company_industry if current_job else '',
#             'Employment Type': current_job.employment_type.value if current_job and current_job.employment_type else '',
#             'Years of Experience': experience_duration,
#             'Total Positions': total_experience_years,
#             'Team Leadership Experience': any(exp.team_size and exp.team_size > 1 for exp in resume_data.work_experience),

#             # Projects and Research
#             'Projects Count': len(resume_data.projects) if resume_data.projects else 0,
#             'GitHub Projects': len([p for p in (resume_data.projects or []) if p.github_url]),
#             'Live Projects': len([p for p in (resume_data.projects or []) if p.live_url]),
#             'Publications Count': len(resume_data.publications) if resume_data.publications else 0,
#             'Patents Count': len(resume_data.patents) if resume_data.patents else 0,

#             # Certifications and Training
#             'Total Certifications': len(resume_data.certifications) if resume_data.certifications else 0,
#             'Active Certifications': len(active_certifications),
#             'Certification Names': safe_join(active_certifications),

#             # Languages
#             'Programming Languages': safe_join([lang.name for lang in (resume_data.languages or []) if lang.is_programming_language]),
#             'Spoken Languages': safe_join([lang.name for lang in (resume_data.languages or []) if not lang.is_programming_language]),
#             'Total Languages': len(resume_data.languages) if resume_data.languages else 0,

#             # Professional Development
#             'Professional Memberships': safe_join(resume_data.professional_memberships),
#             'Conferences Attended': len(resume_data.conferences_attended) if resume_data.conferences_attended else 0,
#             'Speaking Engagements': len(resume_data.speaking_engagements) if resume_data.speaking_engagements else 0,
#             'Volunteer Experience': len(resume_data.volunteer_experience) if resume_data.volunteer_experience else 0,

#             # Personal
#             'Achievements': safe_join(resume_data.achievements),
#             'Personal Interests': safe_join(resume_data.personal_interests),

#             # Preferences
#             'Remote Work Preference': resume_data.remote_work_preference,
#             'Willing to Relocate': resume_data.willing_to_relocate,
#             'Expected Salary': resume_data.expected_salary,
#             'Availability': resume_data.availability,

#             # References
#             'References Available': resume_data.references_available_on_request,
#             'Reference Count': len(resume_data.references) if resume_data.references else 0,
#         }

#         return pd.DataFrame([flat_data])

#     def process_multiple_resumes(self, pdf_files: List[str]) -> pd.DataFrame:
#         """Process multiple resume PDFs and return combined DataFrame"""
#         all_data = []
#         successful_parses = 0

#         for pdf_file in pdf_files:
#             try:
#                 print(f"Processing: {pdf_file}")
#                 resume_data = self.parse_resume(pdf_file)

#                 # Validate the parsed data
#                 if self.validate_parsed_data(resume_data):
#                     flat_df = self.create_flat_dataframe(resume_data)
#                     flat_df['Source_File'] = pdf_file  # Add source tracking
#                     all_data.append(flat_df)
#                     successful_parses += 1
#                     print(f"✅ Successfully parsed: {resume_data.full_name}")
#                 else:
#                     print(f"❌ Validation failed for: {pdf_file}")

#             except Exception as e:
#                 print(f"❌ Error processing {pdf_file}: {e}")
#                 continue
            
#         print(f"\n📊 Summary: {successful_parses}/{len(pdf_files)} resumes parsed successfully")

#         if all_data:
#             final_df = pd.concat(all_data, ignore_index=True)
#             return final_df
#         else:
#             return pd.DataFrame()

#     def save_results(self, data: dict, output_format: str = 'json', custom_filename: str = None):
#         """Save parsed results to appropriate folder with unique naming"""
        
#         # Create directories if they don't exist
#         json_dir = "data/json_output"
#         csv_dir = "data/csv_output"
#         os.makedirs(json_dir, exist_ok=True)
#         os.makedirs(csv_dir, exist_ok=True)
        
#         # Generate filename based on full name or unique ID
#         if custom_filename:
#             base_name = custom_filename
#         else:
#             # Try to use full name, fallback to unique ID
#             full_name = data.get('full_name')
#             if full_name:
#                 # Clean the name for filename (remove special characters)
#                 base_name = "".join(c for c in full_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
#                 base_name = base_name.replace(' ', '_')
#             else:
#                 # Generate unique ID if no name available
#                 base_name = f"resume_{uuid.uuid4().hex[:8]}"
        
#         # Add timestamp for uniqueness
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
#         try:
#             if output_format == 'json':
#                 filename = f"{base_name}_{timestamp}.json"
#                 filepath = os.path.join(json_dir, filename)
#                 with open(filepath, 'w', encoding='utf-8') as f:
#                     json.dump(data, f, indent=2, ensure_ascii=False)
    
#             elif output_format == 'csv':
#                 filename = f"{base_name}_{timestamp}.csv"
#                 filepath = os.path.join(csv_dir, filename)
#                 df = self.create_flat_dataframe(data)
#                 df.to_csv(filepath, index=False, encoding='utf-8')
    
#             else:
#                 raise ValueError("Supported formats: 'json', 'csv'")
    
#             print(f"Results saved to: {filepath}")
#             return filepath
    
#         except Exception as e:
#             print(f"❌ Error saving results: {e}")
#             return None
        
    
#     def validate_parsed_data(self, resume_data) -> bool:
#         """Enhanced validation for parsed resume data"""
#         try:
#             # Handle both dict and ResumeData object
#             if isinstance(resume_data, dict):
#                 full_name = resume_data.get('full_name')
#                 email = resume_data.get('email')
#                 skills = resume_data.get('skills', [])
#                 work_experience = resume_data.get('work_experience', [])
#                 education = resume_data.get('education', [])
#             else:
#                 full_name = resume_data.full_name
#                 email = resume_data.email
#                 skills = resume_data.skills
#                 work_experience = resume_data.work_experience
#                 education = resume_data.education
    
#             # Check required fields
#             required_checks = [
#                 full_name and full_name != "Unknown",
#                 email and "@" in email and email != "unknown@example.com",
#                 len(skills) > 0,
#                 len(work_experience) > 0 or len(education) > 0
#             ]
    
#             # Additional quality checks - FIXED for string list
#             quality_checks = [
#                 # Skills are now strings, not objects
#                 any(skill for skill in skills if isinstance(skill, str) and skill.strip()),
#                 # Check if work experience has meaningful data
#                 not work_experience or any(
#                     exp.get('job_title') if isinstance(exp, dict) else exp.job_title 
#                     for exp in work_experience
#                 ),
#                 # Check if education has meaningful data
#                 not education or any(
#                     edu.get('degree') if isinstance(edu, dict) else edu.degree 
#                     for edu in education
#                 )
#             ]
    
#             basic_valid = all(required_checks)
#             quality_valid = all(quality_checks)
    
#             if basic_valid and quality_valid:
#                 return True
#             else:
#                 print(f"Validation issues - Basic: {basic_valid}, Quality: {quality_valid}")
#                 return False
    
#         except Exception as e:
#             print(f"Validation error: {e}")
#             return False
    

# def main():
#     """Example usage of the ResumeParser with enhanced error handling"""
#     try:
#         # Initialize parser
#         parser = ResumeParser()
        
#         # Parse single resume
#         pdf_path = "Shahnawaz_AI+ML_Resume_2025.pdf"
#         print(f"📄 Parsing resume: {pdf_path}")
        
#         resume_data = parser.parse_resume(pdf_path)
        
#         # Validate results
#         if parser.validate_parsed_data(resume_data):
#             print(f"✅ Successfully parsed resume for: {resume_data.full_name}")
            
#             # Print summary
#             print(f"\n📋 Resume Summary:")
#             print(f"   Name: {resume_data.full_name}")
#             print(f"   Email: {resume_data.email}")
#             print(f"   Title: {resume_data.professional_title}")
#             print(f"   Skills: {len(resume_data.skills)}")
#             print(f"   Experience: {len(resume_data.work_experience)} positions")
#             print(f"   Education: {len(resume_data.education)} entries")
#             print(f"   Projects: {len(resume_data.projects) if resume_data.projects else 0}")
            
#             # Create DataFrame for analysis
#             df = parser.create_flat_dataframe(resume_data)
#             print(f"\n📊 Flattened data columns: {len(df.columns)}")
            
#             # Save results
#             json_path = parser.save_results(resume_data, 'json')
#             csv_path = parser.save_results(resume_data, 'csv')
            
#             print(f"\n💾 Files saved:")
#             print(f"   JSON: {json_path}")
#             print(f"   CSV: {csv_path}")
            
#         else:
#             print("❌ Resume parsing failed validation")
        
#         # Example: Process multiple resumes
#         print(f"\n📚 For batch processing, use:")
#         print(f"   pdf_files = ['resume1.pdf', 'resume2.pdf', 'resume3.pdf']")
#         print(f"   batch_df = parser.process_multiple_resumes(pdf_files)")
#         print(f"   batch_df.to_csv('all_resumes_analysis.csv', index=False)")
        
#     except Exception as e:
#         print(f"❌ Error in main: {e}")


# if __name__ == "__main__":
#     main()

import json
import time
import re
import os
import uuid
import pandas as pd
import fitz
from dotenv import load_dotenv
from datetime import date 
from enum import Enum
from PyPDF2 import PdfReader
from pydantic import BaseModel, Field, HttpUrl, EmailStr, field_validator
from typing import List, Optional, Dict, Any
from datetime import date as DateType
from datetime import datetime

from google import genai
from google.genai import types

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import HumanMessage

class EmploymentType(str, Enum):
    FULL_TIME = "full_time"
    PART_TIME = "part_time"
    CONTRACT = "contract"
    INTERNSHIP = "internship"
    FREELANCE = "freelance"


class EducationLevel(str, Enum):
    HIGH_SCHOOL = "high_school"
    DIPLOMA = "diploma"
    BACHELOR = "bachelor"
    MASTER = "master"
    DOCTORATE = "doctorate"
    CERTIFICATE = "certificate"


class WorkExperience(BaseModel):
    job_title: str = Field(description="Job title/position")
    company_name: str = Field(description="Company name")
    location: Optional[str] = Field(default=None, description="Work location")
    employment_type: Optional[EmploymentType] = Field(default=None, description="Type of employment")
    start_date: Optional[date] = Field(default=None, description="Start date")
    end_date: Optional[date] = Field(default=None, description="End date (None if current)")
    is_current: bool = Field(default=False, description="Is this current position")
    description: Optional[str] = Field(default=None, description="Job description")
    key_responsibilities: Optional[List[str]] = Field(default=None, description="Key responsibilities")

class Education(BaseModel):
    degree: str = Field(description="Degree name")
    institution: str = Field(description="School/university name")
    field_of_study: Optional[str] = Field(default=None, description="Major/field of study")
    education_level: Optional[EducationLevel] = Field(default=None, description="Level of education")
    start_date: Optional[date] = Field(default=None, description="Start date")
    end_date: Optional[date] = Field(default=None, description="Graduation date")
    gpa: Optional[float] = Field(default=None, description="GPA score")

    @field_validator('gpa', mode='before')
    @classmethod
    def parse_gpa(cls, v):
        """Convert GPA strings like '8.65/10' or '84%' to float"""
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            v = v.strip().replace('%', '')
            if '/' in v:
                try:
                    return float(v.split('/')[0].strip())
                except:
                    return None
            try:
                return float(v)
            except:
                return None
        return None


class Project(BaseModel):
    name: str = Field(alias="project_name", description="Project name")
    description: str = Field(description="Project description")
    github_url: Optional[HttpUrl] = Field(default=None, description="GitHub repository URL")
    project_url: Optional[HttpUrl] = Field(default=None, description="Live project URL or demo link")

    class Config:
        populate_by_name = True 

class Certification(BaseModel):
    name: str = Field(description="Certification name")
    issuing_organization: Optional[str] = Field(description="Organization that issued the certification")
    issue_date: Optional[date] = Field(default=None, description="Date issued")
    credential_url: Optional[HttpUrl] = Field(default=None, description="Verification URL")


class ResumeData(BaseModel):
    """Streamlined resume data model for AI recruitment"""
    
    # Basic Information (Required)
    full_name: str = Field(default="Anonymous Candidate")
    email: str = Field(default="anonymous@system.generated")
    phone_number: Optional[str] = None
    location: Optional[str] = Field(default=None, description="Current location/address")
    
    # Professional Identity
    professional_title: Optional[str] = Field(default=None, description="Current job title or desired position")
    professional_summary: Optional[str] = Field(default=None, description="Professional summary/objective")
    
    # Online Presence
    linkedin_url: Optional[HttpUrl] = Field(default=None, description="LinkedIn profile URL")
    github_url: Optional[HttpUrl] = Field(default=None, description="GitHub profile URL")
    portfolio_url: Optional[HttpUrl] = Field(default=None, description="Portfolio website URL")

    # Core Sections (Required)
    skills: List[str] = Field(description="List of skills")
    work_experience: List[WorkExperience] = Field(description="Work experience history")
    education: List[Education] = Field(description="Education history")
    
    # Optional Sections
    projects: Optional[List[Project]] = Field(default=None, description="Projects")
    certifications: Optional[List[Certification]] = Field(default=None, description="Certifications")
    languages: Optional[List[str]] = Field(default=None, description="Spoken languages (NOT programming)")
    achievements: Optional[List[str]] = Field(default=None, description="Awards and notable achievements")


class RegexExtractor:
    """
    PRODUCTION-READY regex extractor with improved patterns.
    Handles edge cases, false positives, and international formats.
    """
    
    def __init__(self):
        # ==================== EMAIL (IMPROVED) ====================
        # Uses lookarounds to avoid trailing punctuation
        # Handles complex TLDs like .co.in, .museum
        self.email_pattern = re.compile(
            r'(?i)(?<![A-Z0-9._%+-])([A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,})(?![A-Z0-9._%+-])'
        )
        
        # ==================== PHONE (IMPROVED) ====================
        # Ordered from specific to generic to minimize false positives
        self.phone_patterns = [
            # Indian format: +91 98765 43210 or 9876543210 (starts with 6-9)
            re.compile(r'\b(?:\+?91[\s-]?)?[6-9]\d{9}\b'),
            
            # US format: (212) 555-0189 or 212-555-0189
            re.compile(r'\b(?:\(\d{3}\)\s*\d{3}[-.]?\d{4}|\d{3}[-.]?\d{3}[-.]?\d{4})\b'),
            
            # E.164-like international: +1234567890 (10-15 digits)
            re.compile(r'\b\+?[1-9]\d{9,14}\b'),
            
            # Generic international (use as last resort)
            re.compile(r'\b\+?\d{1,3}[\s-]?\(?\d{1,4}\)?(?:[\s-]?\d{2,4}){2,4}\b')
        ]
        
        # ==================== LINKEDIN (IMPROVED) ====================
        # Personal profiles only (/in/ and /pub/)
        self.linkedin_pattern = re.compile(
            r'(?i)\b(?:https?://)?(?:[a-z]{2,3}\.)?linkedin\.com/(?:in|pub)/[A-Za-z0-9%._-]+/?(?:\?[^\s]*)?\b'
        )
        
        # ==================== GITHUB (IMPROVED) ====================
        # User/org profile with 39-char limit
        self.github_user_pattern = re.compile(
            r'(?i)\b(?:https?://)?(?:www\.)?github\.com/[A-Za-z0-9-]{1,39}/?(?:\?[^\s]*)?\b'
        )
        
        # Repository pattern
        self.github_repo_pattern = re.compile(
            r'(?i)\b(?:https?://)?(?:www\.)?github\.com/[A-Za-z0-9-]{1,39}/[A-Za-z0-9._-]+/?(?:\?[^\s]*)?\b'
        )
        
        # ==================== PORTFOLIO (IMPROVED) ====================
        # HTTP(S) URLs only
        self.portfolio_pattern = re.compile(
            r'(?i)\bhttps?://(?:www\.)?[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?:/[^\s]*)?\b'
        )
        
        # ==================== LOCATION (IMPROVED) ====================
        # Labeled location: "Location: Bengaluru, Karnataka, India"
        self.location_labeled_pattern = re.compile(
            r'(?im)^(?:Location|Address|Based in|City|State|Country)[:\s-]*([A-Z][A-Za-z .,&-]+(?:,\s*[A-Z][A-Za-z .,&-]+){0,3}(?:,\s*\d{4,10})?)$'
        )
        
        # Unlabeled location: "Bengaluru, Karnataka, India"
        self.location_unlabeled_pattern = re.compile(
            r'(?m)^\s*([A-Z][A-Za-z .,&()-]+(?:,\s*[A-Z][A-Za-z .,&()-]+){1,3})\s*$'
        )
        
        # Known location keywords for validation
        self.location_keywords = {
            'india', 'usa', 'uk', 'canada', 'australia', 'singapore',
            'bangalore', 'bengaluru', 'mumbai', 'delhi', 'pune', 'hyderabad',
            'chennai', 'kolkata', 'ahmedabad', 'gurgaon', 'noida',
            'new york', 'san francisco', 'london', 'tokyo', 'paris'
        }
    
    def extract_email(self, text: str) -> Optional[str]:
        """Extract email with improved validation"""
        match = self.email_pattern.search(text)
        if match:
            email = match.group(1)
            # Validate: must have valid TLD and not end with dot
            if '.' in email.split('@')[1] and not email.endswith('.'):
                return email
        return None
    
    def extract_phone(self, text: str) -> Optional[str]:
        """Extract phone with improved validation and false positive filtering"""
        for pattern in self.phone_patterns:
            match = pattern.search(text)
            if match:
                phone = match.group(0)
                
                # FILTER: Avoid dates like "2024-10-23"
                if '-' in phone:
                    parts = phone.split('-')
                    if len(parts) == 3 and all(p.isdigit() for p in parts):
                        if len(parts[0]) == 4 and len(parts[1]) <= 2 and len(parts[2]) <= 2:
                            continue  # Skip this match, likely a date
                
                # Validate length
                digits_only = re.sub(r'\D', '', phone)
                if 10 <= len(digits_only) <= 15:
                    return phone
        
        return None
    
    def extract_linkedin(self, text: str) -> Optional[str]:
        """Extract LinkedIn URL with normalization"""
        match = self.linkedin_pattern.search(text)
        if match:
            url = match.group(0)
            if not url.startswith('http'):
                url = 'https://' + url
            # Clean up
            url = url.rstrip('/')
            if '?' in url:
                url = url.split('?')[0]
            return url
        return None
    
    def extract_github(self, text: str) -> Optional[str]:
        """Extract GitHub URL with validation"""
        # Try repo pattern first (more specific)
        match = self.github_repo_pattern.search(text)
        if match:
            url = match.group(0)
        else:
            # Try user pattern
            match = self.github_user_pattern.search(text)
            if match:
                url = match.group(0)
            else:
                return None
        
        # Normalize
        if not url.startswith('http'):
            url = 'https://' + url
        url = url.rstrip('/')
        if '?' in url:
            url = url.split('?')[0]
        
        # FILTER: Exclude invalid paths
        invalid_paths = ['/topics/', '/marketplace/', '/explore/', '/features/']
        if any(path in url.lower() for path in invalid_paths):
            return None
        
        return url
    
    def extract_portfolio(self, text: str) -> Optional[str]:
        """Extract portfolio URL with filtering"""
        matches = self.portfolio_pattern.finditer(text)
        
        for match in matches:
            url = match.group(0).rstrip('/')
            url_lower = url.lower()
            
            # FILTER: Exclude LinkedIn and GitHub
            if 'linkedin.com' in url_lower or 'github.com' in url_lower:
                continue
            
            # FILTER: Exclude common non-portfolio sites
            excluded = [
                'google.com', 'facebook.com', 'twitter.com', 'instagram.com',
                'youtube.com', 'stackoverflow.com', 'medium.com',
                'indeed.com', 'naukri.com', 'monster.com'
            ]
            
            if not any(domain in url_lower for domain in excluded):
                return url
        
        return None
    
    def extract_location(self, text: str) -> Optional[str]:
        """Extract location with improved patterns"""
        # Try labeled pattern first
        match = self.location_labeled_pattern.search(text)
        if match:
            location = match.group(1).strip()
            return self._validate_location(location)
        
        # Try unlabeled pattern
        matches = self.location_unlabeled_pattern.finditer(text)
        for match in matches:
            location = match.group(1).strip()
            validated = self._validate_location(location)
            if validated:
                return validated
        
        return None
    
    def _validate_location(self, location: str) -> Optional[str]:
        """Validate location string"""
        if not location or len(location) > 100:
            return None
        
        # Must have comma (city, state/country format)
        if ',' not in location:
            return None
        
        # Check for known location keywords
        location_lower = location.lower()
        if any(kw in location_lower for kw in self.location_keywords):
            return location
        
        # If 2-4 components, likely valid
        components = [c.strip() for c in location.split(',')]
        if 2 <= len(components) <= 4:
            if all(c and c[0].isupper() for c in components):
                return location
        
        return None
    
    def extract_name(self, text: str) -> Optional[str]:
        """Extract name with improved validation"""
        lines = text.split('\n')
        
        for i, line in enumerate(lines[:10]):
            line = line.strip()
            
            if not line:
                continue
            
            # Skip headers
            skip_kw = [
                'resume', 'cv', 'curriculum', 'email', 'phone',
                'address', 'objective', 'summary', 'experience',
                'education', 'skills', 'projects'
            ]
            
            if any(kw in line.lower() for kw in skip_kw):
                continue
            
            words = line.split()
            
            # Name: 2-4 words, capitalized, 5-50 chars
            if 2 <= len(words) <= 4:
                if all(w[0].isupper() for w in words if w):
                    # Valid name characters
                    if re.match(r"^[A-Z][a-z]+(?:[\s'-][A-Z][a-z]+){1,3}$", line):
                        # Not all caps
                        if not line.isupper():
                            if 5 <= len(line) <= 50:
                                return line
        
        return None


class ResumeParser:
    """AI-powered resume parser using hybrid approach (Regex + Gemini)"""
    
    def __init__(self, api_key: str = None):
        """Initialize the resume parser"""
        load_dotenv()
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")

        if not self.api_key:
            raise ValueError("Google API key not found. Set GOOGLE_API_KEY environment variable.")
        
        # Initialize regex extractor
        self.regex_extractor = RegexExtractor()
        
        # Initialize Gemini LLM for complex extraction
        self.llm = ChatGoogleGenerativeAI(
            temperature=0, 
            model='gemini-2.0-flash',
            google_api_key=self.api_key,
            max_output_tokens= 12288,
            streaming=True
        )
        
        # Complex fields template (only for fields that need AI)
        self.complex_template = PromptTemplate(
            template="""Extract the following COMPLEX fields from this resume. Return ONLY valid JSON

FIELDS TO EXTRACT:
1. skills: Array of technical skills, tools, frameworks, languages
2. work_experience: Array of work history with detailed information
3. education: Array of educational qualifications
4. projects: Array of projects (if any)
5. certifications: Array of certifications (if any)
6. professional_summary: Brief professional summary or objective
7. professional_title: Current or most recent job title
8. languages: Array of spoken/written languages (NOT programming languages)
9. achievements: Array of notable achievements or awards

WORK EXPERIENCE FORMAT:
{{
  "job_title": "string",
  "company_name": "string", 
  "location": "string or null",
  "employment_type": "full_time|part_time|contract|internship|freelance",
  "start_date": "YYYY-MM-DD or null",
  "end_date": "YYYY-MM-DD or null",
  "is_current": boolean,
  "description": "string or null",
  "key_responsibilities": ["array"] or null
}}

EDUCATION FORMAT:
{{
  "degree": "string",
  "institution": "string",
  "field_of_study": "string or null",
  "education_level": "bachelor|master|doctorate|diploma|certificate",
  "start_date": "YYYY-MM-DD or null",
  "end_date": "YYYY-MM-DD or null",
  "gpa": number or null
}}

PROJECTS FORMAT:
{{
  "name": "Project Name",
  "description": "Description",
  "github_url": "https://github.com/user/repo" or null,
  "project_url": "https://..." or null
}}

CERTIFICATIONS FORMAT:
{{
  "name": "Certification Name",
  "issuing_organization": "Organization",
  "issue_date": "YYYY-MM-DD or null",
  "credential_url": "https://..." or null
}}

Rules

-Output only a single valid JSON object. No markdown/code fences/comments.

-Use null for unknown values and [] for empty arrays.

-Extract ONLY what appears in the resume. Do not infer or add related terms.

-Do NOT include job titles in skills.

-Skills must match the resume text as written (preserve original casing/spelling).

-Dates: if exact day is provided, use it. If month/year only, set day to 01 (e.g., June 2025 → 2025-06-01). If only year, use YYYY-01-01. If unknown, use null.

-For employment_type, set one of the allowed values if explicitly stated; otherwise null.

-Keep key_responsibilities as short bullet-like strings if present; else null.

-Ensure valid JSON: double-quoted keys/strings, no trailing commas.


# RESUME TEXT:
# {resume_text}

# Return ONLY the JSON object:""",
            input_variables=["resume_text"]
        )
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        try:
            text = ""
            with fitz.open(pdf_path) as pdf:
                for page in pdf:
                    text += page.get_text()
            return text.strip()
        except Exception as e:
            print(f"❌ Error extracting text from PDF: {e}")
            return ""
    
    def extract_easy_fields(self, text: str) -> Dict[str, Any]:
        """Extract simple fields using regex"""
        print("🔍 Extracting easy fields with regex...")
        
        easy_fields = {
            'full_name': self.regex_extractor.extract_name(text),
            'email': self.regex_extractor.extract_email(text),
            'phone_number': self.regex_extractor.extract_phone(text),
            'linkedin_url': self.regex_extractor.extract_linkedin(text),
            'github_url': self.regex_extractor.extract_github(text),
            'portfolio_url': self.regex_extractor.extract_portfolio(text),
            'location': self.regex_extractor.extract_location(text),
        }
        
        # Log what was found
        print(f"   ✓ Name: {easy_fields['full_name'] or 'Not found'}")
        print(f"   ✓ Email: {easy_fields['email'] or 'Not found'}")
        print(f"   ✓ Phone: {easy_fields['phone_number'] or 'Not found'}")
        print(f"   ✓ LinkedIn: {'Found' if easy_fields['linkedin_url'] else 'Not found'}")
        print(f"   ✓ GitHub: {'Found' if easy_fields['github_url'] else 'Not found'}")
        
        return easy_fields
    
    def extract_complex_fields(self, text: str, max_retries: int = 2) -> Dict[str, Any]:
        """Extract complex fields using Gemini AI with retry logic and JSON cleanup"""
        print("🤖 Extracting complex fields with Gemini AI...")

        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    print(f"   🔄 Retry attempt {attempt + 1}/{max_retries}")

                # Create prompt
                prompt = self.complex_template.format(resume_text=text)

                # Call Gemini
                response = self.llm.invoke(prompt)
                response_text = response.content.strip()

                # Clean response (remove markdown code blocks if present)
                if response_text.startswith('```'):
                    response_text = response_text.split('```')[1]
                    if response_text.startswith('json'):
                        response_text = response_text[4:]
                    response_text = response_text.strip()

                # Additional cleanup: Fix incomplete/truncated JSON
                # Check for unbalanced braces
                open_braces = response_text.count('{')
                close_braces = response_text.count('}')

                if open_braces > close_braces:
                    # Response was truncated, find last complete closing brace
                    last_brace = response_text.rfind('}')
                    if last_brace != -1:
                        response_text = response_text[:last_brace + 1]
                        print("   ⚠️  Detected truncated response, cleaned up JSON")

                # Check for unterminated strings (common issue)
                if response_text.count('"') % 2 != 0:
                    # Odd number of quotes means unterminated string
                    # Try to find last complete field
                    last_complete = max(
                        response_text.rfind('},'),
                        response_text.rfind('],'),
                        response_text.rfind('"}')
                    )
                    if last_complete != -1:
                        # Cut at last complete field and close JSON
                        response_text = response_text[:last_complete + 1]
                        # Add closing braces as needed
                        while response_text.count('{') > response_text.count('}'):
                            response_text += '}'
                        print("   ⚠️  Fixed unterminated string in response")

                # Parse JSON
                complex_data = json.loads(response_text)

                print(f"   ✓ Skills: {len(complex_data.get('skills', []))} found")
                print(f"   ✓ Work Experience: {len(complex_data.get('work_experience', []))} entries")
                print(f"   ✓ Education: {len(complex_data.get('education', []))} entries")
                print(f"   ✓ Projects: {len(complex_data.get('projects', []))} found")

                return complex_data

            except json.JSONDecodeError as e:
                print(f"❌ JSON parsing error (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    print("   🔄 Retrying with fresh request...")
                    time.sleep(1)  # Brief delay before retry
                    continue
                print(f"   Response preview: {response_text[:500]}")
                return self._get_empty_complex_fields()

            except Exception as e:
                print(f"❌ Error extracting complex fields (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    print("   🔄 Retrying...")
                    time.sleep(1)
                    continue
                return self._get_empty_complex_fields()
    
        # Should never reach here, but just in case
        return self._get_empty_complex_fields()
    
    def _get_empty_complex_fields(self) -> Dict[str, Any]:
        """Return empty structure for complex fields"""
        return {
            'skills': [],
            'work_experience': [],
            'education': [],
            'projects': None,
            'certifications': None,
            'professional_summary': None,
            'professional_title': None,
            'languages': None,
            'achievements': None
        }
    
    def parse_resume(self, pdf_path: str, max_retries: int = 2) -> ResumeData:
        """
        Parse resume using hybrid approach:
        1. Extract easy fields with regex
        2. Extract complex fields with Gemini
        3. Merge and validate
        """
        print(f"\n{'='*60}")
        print(f"📄 Processing: {pdf_path}")
        print(f"{'='*60}")
        
        # Extract text from PDF
        resume_text = self.extract_text_from_pdf(pdf_path)
        if not resume_text:
            raise ValueError("Could not extract text from PDF")
        
        print(f"📝 Extracted {len(resume_text)} characters from PDF")
        
        # Step 1: Extract easy fields with regex
        easy_fields = self.extract_easy_fields(resume_text)
        
        # Step 2: Extract complex fields with Gemini
        complex_fields = self.extract_complex_fields(resume_text)
        
        # Step 3: Merge both results
        print("\n🔄 Merging extracted data...")
        merged_data = {**easy_fields, **complex_fields}
        
        # Step 4: Clean and validate
        merged_data = self._clean_resume_data(merged_data)
        
        # Step 5: Create ResumeData object
        try:
            resume_data = ResumeData(**merged_data)
            print("✅ Successfully created ResumeData object")
            return resume_data
        except Exception as e:
            print(f"❌ Pydantic validation error: {e}")
            print(f"Attempting fallback with defaults...")
            
            # Provide defaults for required fields
            if not merged_data.get('full_name'):
                merged_data['full_name'] = "Anonymous Candidate"
            if not merged_data.get('email'):
                merged_data['email'] = "anonymous@system.generated"
            if not merged_data.get('skills'):
                merged_data['skills'] = []
            if not merged_data.get('work_experience'):
                merged_data['work_experience'] = []
            if not merged_data.get('education'):
                merged_data['education'] = []

            # Fix education entries with missing required fields
            if merged_data.get('education'):
                for edu in merged_data['education']:
                    if not edu.get('degree'):
                        edu['degree'] = "Not Specified"
                    if not edu.get('institution'):
                        edu['institution'] = "Not Specified"

            # Fix work experience entries with missing required fields
            if merged_data.get('work_experience'):
                for exp in merged_data['work_experience']:
                    if not exp.get('job_title'):
                        exp['job_title'] = "Not Specified"
                    if not exp.get('company_name'):
                        exp['company_name'] = "Not Specified"
            
            resume_data = ResumeData(**merged_data)
            return resume_data
    
    def _clean_resume_data(self, data: dict) -> dict:
        """Clean and prepare data for Pydantic validation"""
        # Convert GPA strings to floats
        if 'education' in data and data['education']:
            for edu in data['education']:
                if 'gpa' in edu and edu['gpa'] is not None:
                    if isinstance(edu['gpa'], str):
                        gpa_str = edu['gpa'].strip().replace('%', '')
                        if '/' in gpa_str:
                            try:
                                edu['gpa'] = float(gpa_str.split('/')[0].strip())
                            except:
                                edu['gpa'] = None
                        else:
                            try:
                                edu['gpa'] = float(gpa_str)
                            except:
                                edu['gpa'] = None
                    elif not isinstance(edu['gpa'], (int, float)):
                        edu['gpa'] = None
        
        # Convert language objects to simple strings
        if 'languages' in data and data['languages']:
            cleaned_languages = []
            for lang in data['languages']:
                if isinstance(lang, dict):
                    cleaned_languages.append(lang.get('name', str(lang)))
                elif isinstance(lang, str):
                    cleaned_languages.append(lang)
            data['languages'] = cleaned_languages if cleaned_languages else None
        
        return data
    
    def create_flat_dataframe(self, resume_data) -> pd.DataFrame:
        """Convert ResumeData to flat DataFrame for analysis"""
        if isinstance(resume_data, ResumeData):
            data_dict = resume_data.model_dump()
        else:
            data_dict = resume_data
        
        flat_data = {
            'full_name': data_dict.get('full_name'),
            'email': data_dict.get('email'),
            'phone_number': data_dict.get('phone_number'),
            'location': data_dict.get('location'),
            'professional_title': data_dict.get('professional_title'),
            'linkedin_url': data_dict.get('linkedin_url'),
            'github_url': data_dict.get('github_url'),
            'portfolio_url': data_dict.get('portfolio_url'),
            'skills_count': len(data_dict.get('skills', [])),
            'skills': ', '.join(data_dict.get('skills', [])),
            'work_experience_count': len(data_dict.get('work_experience', [])),
            'education_count': len(data_dict.get('education', [])),
            'projects_count': len(data_dict.get('projects', [])) if data_dict.get('projects') else 0,
            'certifications_count': len(data_dict.get('certifications', [])) if data_dict.get('certifications') else 0,
        }
        
        return pd.DataFrame([flat_data])
    
    def save_results(self, data, output_format: str = 'json', custom_filename: str = None):
        """Save parsed results to appropriate folder"""
        json_dir = "data/json_output"
        csv_dir = "data/csv_output"
        os.makedirs(json_dir, exist_ok=True)
        os.makedirs(csv_dir, exist_ok=True)
        
        # Convert to dict if needed
        if isinstance(data, ResumeData):
            data_dict = data.model_dump()
        else:
            data_dict = data
        
        # Generate filename
        if custom_filename:
            base_name = custom_filename
        else:
            full_name = data_dict.get('full_name')
            if full_name and full_name != "Anonymous Candidate":
                base_name = "".join(c for c in full_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
                base_name = base_name.replace(' ', '_')
            else:
                base_name = f"resume_{uuid.uuid4().hex[:8]}"
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            if output_format == 'json':
                filename = f"{base_name}_{timestamp}.json"
                filepath = os.path.join(json_dir, filename)
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(data_dict, f, indent=2, ensure_ascii=False, default=str)
            
            elif output_format == 'csv':
                filename = f"{base_name}_{timestamp}.csv"
                filepath = os.path.join(csv_dir, filename)
                df = self.create_flat_dataframe(data_dict)
                df.to_csv(filepath, index=False, encoding='utf-8')
            
            else:
                raise ValueError("Supported formats: 'json', 'csv'")
            
            print(f"💾 Results saved to: {filepath}")
            return filepath
        
        except Exception as e:
            print(f"❌ Error saving results: {e}")
            return None
    
    def validate_parsed_data(self, resume_data) -> bool:
        """Enhanced validation for parsed resume data"""
        try:
            if isinstance(resume_data, dict):
                full_name = resume_data.get('full_name')
                email = resume_data.get('email')
                skills = resume_data.get('skills', [])
                work_experience = resume_data.get('work_experience', [])
                education = resume_data.get('education', [])
            else:
                full_name = resume_data.full_name
                email = resume_data.email
                skills = resume_data.skills
                work_experience = resume_data.work_experience
                education = resume_data.education
            
            required_checks = [
                full_name and full_name != "Anonymous Candidate",
                email and "@" in email and email != "anonymous@system.generated",
                len(skills) > 0,
                len(work_experience) > 0 or len(education) > 0
            ]
            
            quality_checks = [
                any(skill for skill in skills if isinstance(skill, str) and skill.strip()),
                not work_experience or any(
                    exp.get('job_title') if isinstance(exp, dict) else exp.job_title 
                    for exp in work_experience
                ),
                not education or any(
                    edu.get('degree') if isinstance(edu, dict) else edu.degree 
                    for edu in education
                )
            ]
            
            basic_valid = all(required_checks)
            quality_valid = all(quality_checks)
            
            if basic_valid and quality_valid:
                return True
            else:
                print(f"⚠️  Validation issues - Basic: {basic_valid}, Quality: {quality_valid}")
                return False
        
        except Exception as e:
            print(f"❌ Validation error: {e}")
            return False


def main():
    """Example usage of the hybrid ResumeParser"""
    try:
        print("🚀 Initializing Hybrid Resume Parser (Regex + Gemini)")
        print("=" * 60)
        
        parser = ResumeParser()
        
        # Parse single resume
        pdf_path = "Shahnawaz_AI+ML_Resume_2025.pdf"
        
        resume_data = parser.parse_resume(pdf_path)
        
        # Validate results
        if parser.validate_parsed_data(resume_data):
            print(f"\n{'='*60}")
            print(f"✅ SUCCESS: Resume parsed and validated")
            print(f"{'='*60}")
            
            # Print summary
            print(f"\n📋 Resume Summary:")
            print(f"   Name: {resume_data.full_name}")
            print(f"   Email: {resume_data.email}")
            print(f"   Phone: {resume_data.phone_number or 'N/A'}")
            print(f"   Title: {resume_data.professional_title or 'N/A'}")
            print(f"   LinkedIn: {'✓' if resume_data.linkedin_url else '✗'}")
            print(f"   GitHub: {'✓' if resume_data.github_url else '✗'}")
            print(f"   Skills: {len(resume_data.skills)}")
            print(f"   Experience: {len(resume_data.work_experience)} positions")
            print(f"   Education: {len(resume_data.education)} entries")
            print(f"   Projects: {len(resume_data.projects) if resume_data.projects else 0}")
            
            # Save results
            json_path = parser.save_results(resume_data, 'json')
            csv_path = parser.save_results(resume_data, 'csv')
            
            print(f"\n{'='*60}")
            print(f"💾 Files saved successfully!")
            print(f"{'='*60}")
            
        else:
            print("\n❌ Resume parsing failed validation")
        
    except Exception as e:
        print(f"\n❌ Error in main: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()