# from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Depends
# from fastapi.responses import JSONResponse
# from pydantic import BaseModel
# from typing import Optional, Dict, Any
# import os
# import tempfile
# import shutil
# from datetime import datetime
# from dotenv import load_dotenv

# # Import your existing ResumeParser class
# # Make sure the resume parser file is in the same directory or adjust the import
# from resume_parser import ResumeParser, ResumeData

# # Load environment variables
# load_dotenv()

# app = FastAPI(
#     title="AI Resume Parser API",
#     description="Extract structured data from PDF resumes using AI",
#     version="1.0.0"
# )

# # Initialize parser globally
# parser = None

# # API Key from environment variable
# API_KEY = os.getenv("API_KEY", "default-dev-key-change-in-production")

# # Security: API Key verification
# async def verify_api_key(x_api_key: str = Header(..., description="API Key for authentication")):
#     """Verify API key from request header"""
#     if x_api_key != API_KEY:
#         raise HTTPException(
#             status_code=403,
#             detail="Invalid or missing API key"
#         )
#     return x_api_key

# @app.on_event("startup")
# async def startup_event():
#     """Initialize the resume parser on startup"""
#     global parser
#     try:
#         parser = ResumeParser()
#         print("✅ Resume Parser initialized successfully")
#         print(f"✅ API Key protection enabled")
#     except Exception as e:
#         print(f"❌ Failed to initialize parser: {e}")
#         raise


# class ParseResponse(BaseModel):
#     """Response model for parsed resume"""
#     success: bool
#     message: str
#     data: Optional[Dict[str, Any]] = None
#     parsing_time: Optional[float] = None
#     error: Optional[str] = None


# class HealthResponse(BaseModel):
#     """Health check response"""
#     status: str
#     timestamp: str
#     parser_initialized: bool


# @app.get("/", response_model=dict)
# async def root():
#     """Root endpoint - No authentication required"""
#     return {
#         "message": "AI Resume Parser API",
#         "version": "1.0.0",
#         "endpoints": {
#             "health": "/health",
#             "parse": "/parse-resume (requires API key)",
#             "docs": "/docs"
#         },
#         "authentication": "Send X-API-Key header with your API key"
#     }


# @app.get("/health", response_model=HealthResponse)
# async def health_check():
#     """Health check endpoint - No authentication required"""
#     return HealthResponse(
#         status="healthy",
#         timestamp=datetime.now().isoformat(),
#         parser_initialized=parser is not None
#     )


# @app.post("/parse-resume", response_model=ParseResponse, dependencies=[Depends(verify_api_key)])
# async def parse_resume(
#     file: UploadFile = File(..., description="PDF resume file to parse")
# ):
#     """
#     Parse a single PDF resume and extract structured information
    
#     - **file**: PDF file containing the resume
    
#     Returns structured resume data including:
#     - Personal information (name, email, phone, location)
#     - Professional summary and title
#     - Work experience
#     - Education
#     - Skills
#     - Projects, certifications, publications (if present)
#     """
    
#     # Validate parser initialization
#     if parser is None:
#         raise HTTPException(
#             status_code=500,
#             detail="Resume parser not initialized. Please restart the server."
#         )
    
#     # Validate file type
#     if not file.filename.lower().endswith('.pdf'):
#         raise HTTPException(
#             status_code=400,
#             detail="Only PDF files are supported"
#         )
    
#     # Create temporary file
#     temp_file_path = None
    
#     try:
#         # Save uploaded file to temporary location
#         with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
#             temp_file_path = temp_file.name
#             shutil.copyfileobj(file.file, temp_file)
        
#         # Parse the resume
#         start_time = datetime.now()
#         resume_data = parser.parse_resume(temp_file_path)
#         parsing_time = (datetime.now() - start_time).total_seconds()
        
#         # Validate parsed data
#         if not parser.validate_parsed_data(resume_data):
#             return ParseResponse(
#                 success=False,
#                 message="Resume parsed but validation failed",
#                 data=resume_data,
#                 parsing_time=parsing_time,
#                 error="Parsed data did not meet quality standards"
#             )
        
#         return ParseResponse(
#             success=True,
#             message=f"Successfully parsed resume for {resume_data.get('full_name', 'Unknown')}",
#             data=resume_data,
#             parsing_time=parsing_time
#         )
        
#     except Exception as e:
#         raise HTTPException(
#             status_code=500,
#             detail=f"Error parsing resume: {str(e)}"
#         )
    
#     finally:
#         # Cleanup temporary file
#         if temp_file_path and os.path.exists(temp_file_path):
#             try:
#                 os.unlink(temp_file_path)
#             except Exception as e:
#                 print(f"Warning: Failed to delete temporary file: {e}")
        
#         # Close uploaded file
#         await file.close()


# @app.post("/parse-resume-detailed", response_model=dict, dependencies=[Depends(verify_api_key)])
# async def parse_resume_detailed(
#     file: UploadFile = File(..., description="PDF resume file to parse"),
#     include_validation: bool = True
# ):
#     """
#     Parse resume with detailed validation results and statistics
    
#     - **file**: PDF file containing the resume
#     - **include_validation**: Include detailed validation results
#     """
    
#     if parser is None:
#         raise HTTPException(
#             status_code=500,
#             detail="Resume parser not initialized"
#         )
    
#     if not file.filename.lower().endswith('.pdf'):
#         raise HTTPException(
#             status_code=400,
#             detail="Only PDF files are supported"
#         )
    
#     temp_file_path = None
    
#     try:
#         with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
#             temp_file_path = temp_file.name
#             shutil.copyfileobj(file.file, temp_file)
        
#         start_time = datetime.now()
#         resume_data = parser.parse_resume(temp_file_path)
#         parsing_time = (datetime.now() - start_time).total_seconds()
        
#         # Create detailed response
#         response = {
#             "success": True,
#             "parsing_time_seconds": parsing_time,
#             "filename": file.filename,
#             "data": resume_data,
#             "statistics": {
#                 "total_skills": len(resume_data.get('skills', [])),
#                 "work_experience_count": len(resume_data.get('work_experience', [])),
#                 "education_count": len(resume_data.get('education', [])),
#                 "projects_count": len(resume_data.get('projects', [])) if resume_data.get('projects') else 0,
#                 "certifications_count": len(resume_data.get('certifications', [])) if resume_data.get('certifications') else 0,
#                 "has_linkedin": bool(resume_data.get('linkedin_url')),
#                 "has_github": bool(resume_data.get('github_url')),
#             }
#         }
        
#         if include_validation:
#             is_valid = parser.validate_parsed_data(resume_data)
#             response["validation"] = {
#                 "passed": is_valid,
#                 "has_required_fields": all([
#                     resume_data.get('full_name'),
#                     resume_data.get('email'),
#                     resume_data.get('skills'),
#                 ])
#             }
        
#         return response
        
#     except Exception as e:
#         raise HTTPException(
#             status_code=500,
#             detail=f"Error parsing resume: {str(e)}"
#         )
    
#     finally:
#         if temp_file_path and os.path.exists(temp_file_path):
#             try:
#                 os.unlink(temp_file_path)
#             except:
#                 pass
#         await file.close()


# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)

from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import os
import tempfile
import shutil
from datetime import datetime
from dotenv import load_dotenv

# Import your existing ResumeParser class
# Make sure the resume parser file is in the same directory or adjust the import
from parser.resume_parser import ResumeParser, ResumeData

# Load environment variables
load_dotenv()

app = FastAPI(
    title="AI Resume Parser API",
    description="Extract structured data from PDF resumes using AI",
    version="1.0.0"
)

# Initialize parser globally
parser = None

# API Key from environment variable
API_KEY = os.getenv("API_KEY", "default-dev-key-change-in-production")

# Security: API Key verification
async def verify_api_key(x_api_key: str = Header(..., description="API Key for authentication")):
    """Verify API key from request header"""
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=403,
            detail="Invalid or missing API key"
        )
    return x_api_key

@app.on_event("startup")
async def startup_event():
    """Initialize the resume parser on startup"""
    global parser
    try:
        parser = ResumeParser()
        print("✅ Resume Parser initialized successfully")
        print(f"✅ API Key protection enabled")
    except Exception as e:
        print(f"❌ Failed to initialize parser: {e}")
        raise


class ParseResponse(BaseModel):
    """Response model for parsed resume"""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    parsing_time: Optional[float] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    parser_initialized: bool


@app.get("/", response_model=dict)
async def root():
    """Root endpoint - No authentication required"""
    return {
        "message": "AI Resume Parser API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "parse": "/parse-resume (requires API key)",
            "docs": "/docs"
        },
        "authentication": "Send X-API-Key header with your API key"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint - No authentication required"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        parser_initialized=parser is not None
    )


@app.post("/parse-resume", response_model=ParseResponse, dependencies=[Depends(verify_api_key)])
async def parse_resume(
    file: UploadFile = File(..., description="PDF resume file to parse")
):
    """
    Parse a single PDF resume and extract structured information
    
    - **file**: PDF file containing the resume
    
    Returns structured resume data including:
    - Personal information (name, email, phone, location)
    - Professional summary and title
    - Work experience
    - Education
    - Skills
    - Projects, certifications, publications (if present)
    """
    
    # Validate parser initialization
    if parser is None:
        raise HTTPException(
            status_code=500,
            detail="Resume parser not initialized. Please restart the server."
        )
    
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )
    
    # Create temporary file
    temp_file_path = None
    
    try:
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file_path = temp_file.name
            shutil.copyfileobj(file.file, temp_file)
        
        # Parse the resume
        start_time = datetime.now()
        resume_data = parser.parse_resume(temp_file_path)
        parsing_time = (datetime.now() - start_time).total_seconds()
        
        # Convert ResumeData to dict if it's a Pydantic model
        if isinstance(resume_data, ResumeData):
            resume_dict = resume_data.model_dump()
        else:
            resume_dict = resume_data
        
        # Validate parsed data
        if not parser.validate_parsed_data(resume_data):
            return ParseResponse(
                success=False,
                message="Resume parsed but validation failed",
                data=resume_dict,
                parsing_time=parsing_time,
                error="Parsed data did not meet quality standards"
            )
        
        return ParseResponse(
            success=True,
            message=f"Successfully parsed resume for {resume_dict.get('full_name', 'Unknown')}",
            data=resume_dict,
            parsing_time=parsing_time
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error parsing resume: {str(e)}"
        )
    
    finally:
        # Cleanup temporary file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                print(f"Warning: Failed to delete temporary file: {e}")
        
        # Close uploaded file
        await file.close()


@app.post("/parse-resume-detailed", response_model=dict, dependencies=[Depends(verify_api_key)])
async def parse_resume_detailed(
    file: UploadFile = File(..., description="PDF resume file to parse"),
    include_validation: bool = True
):
    """
    Parse resume with detailed validation results and statistics
    
    - **file**: PDF file containing the resume
    - **include_validation**: Include detailed validation results
    """
    
    if parser is None:
        raise HTTPException(
            status_code=500,
            detail="Resume parser not initialized"
        )
    
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )
    
    temp_file_path = None
    
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file_path = temp_file.name
            shutil.copyfileobj(file.file, temp_file)
        
        start_time = datetime.now()
        resume_data = parser.parse_resume(temp_file_path)
        parsing_time = (datetime.now() - start_time).total_seconds()
        
        # Convert ResumeData to dict if it's a Pydantic model
        if isinstance(resume_data, ResumeData):
            resume_dict = resume_data.model_dump()
        else:
            resume_dict = resume_data
        
        # Create detailed response
        response = {
            "success": True,
            "parsing_time_seconds": parsing_time,
            "filename": file.filename,
            "data": resume_dict,
            "statistics": {
                "total_skills": len(resume_dict.get('skills', [])),
                "work_experience_count": len(resume_dict.get('work_experience', [])),
                "education_count": len(resume_dict.get('education', [])),
                "projects_count": len(resume_dict.get('projects', [])) if resume_dict.get('projects') else 0,
                "certifications_count": len(resume_dict.get('certifications', [])) if resume_dict.get('certifications') else 0,
                "has_linkedin": bool(resume_dict.get('linkedin_url')),
                "has_github": bool(resume_dict.get('github_url')),
            }
        }
        
        if include_validation:
            is_valid = parser.validate_parsed_data(resume_data)
            response["validation"] = {
                "passed": is_valid,
                "has_required_fields": all([
                    resume_dict.get('full_name'),
                    resume_dict.get('email'),
                    resume_dict.get('skills'),
                ])
            }
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error parsing resume: {str(e)}"
        )
    
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except:
                pass
        await file.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)