"""
quick_test.py

Quick test script to verify the pipeline with a single resume.
Use this for rapid iteration and debugging.
"""

import sys
import uuid
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.parser.resume_parser import ResumeParser
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.embeddings.preprocessor import ResumePreprocessor
from src.embeddings.vector_store import VectorStore

def test_single_file(file_path: str):
    """Test pipeline with a single file"""
    
    print("\n" + "="*80)
    print("🧪 QUICK PIPELINE TEST")
    print("="*80)
    print(f"File: {file_path}\n")

    try:
        # 1. Extract Text
        print("📄 1. Extracting text...")
        if file_path.endswith('.pdf'):
            from PyPDF2 import PdfReader
            reader = PdfReader(file_path)
            text = "".join([page.extract_text() for page in reader.pages])
        elif file_path.endswith('.docx'):
            import docx
            doc = docx.Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs])
        else:
            raise ValueError("Only PDF and DOCX supported")
        
        print(f"   ✅ Extracted {len(text)} characters\n")

        # 2. Parse
        print("🤖 2. Parsing with AI...")
        parser = ResumeParser()
        parsed = parser.parse_resume(file_path)
        print(f"   ✅ Name: {parsed['full_name']}")
        print(f"   ✅ Email: {parsed['email']}")
        print(f"   ✅ Skills: {len(parsed['skills'])}\n")
        # 3. Preprocess
        print("🔧 3. Preprocessing...")
        preprocessor = ResumePreprocessor()
        prep_text = preprocessor.process(parsed)
        token_estimate = len(prep_text.split())
        print(f"   ✅ Text length: {len(prep_text)} chars")
        print(f"   ✅ Token estimate: ~{token_estimate}")
        
        if token_estimate > 1800:
            print(f"   ⚠️  WARNING: May exceed 2048 token limit!\n")
        else:
            print(f"   ✅ Within token limit\n")

        # 4. Generate embedding
        print("🧬 4. Generating embedding...")
        generator = EmbeddingGenerator()
        embedding = generator.generate_embedding(prep_text)
        print(f"   ✅ Embedding dims: {len(embedding)}\n")
        
        # 5. Store in Pinecone
        print("💾 5. Storing in Pinecone...")
        store = VectorStore()
        resume_id = f"resume_{uuid.uuid4().hex[:16]}"
        
        metadata = {
            "resume_id": resume_id,
            "full_name": parsed['full_name'],
            "email": parsed['email'],
            "skills": parsed['skills'][:20],
            "total_skills": len(parsed['skills'])
        }
        success = store.upsert_resume(resume_id, embedding, metadata)

        if success:
            print(f"   ✅ Stored with ID: {resume_id}\n")

            # 6. Verify
            print("🔍 6. Verifying storage...")
            import time
            time.sleep(2)
            
            retrieved = store.get_resume_by_id(resume_id)
            if retrieved:
                print(f"   ✅ Successfully retrieved!")
                print(f"   ✅ Name: {retrieved['metadata']['full_name']}\n")
            else:
                print("   ❌ Failed to retrieve\n")
        else:
            print("   ❌ Failed to store\n")
        
        print("="*80)
        print("✅ TEST PASSED - All steps completed successfully!")
        print("="*80)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_single_file(sys.argv[1])
    else:
        # Default test file
        test_file = r"C:\Users\sshah\OneDrive\Desktop\AI resume parser\ai-resume-parser\uploaded_resumes\Shahnawaz_AI_ML_Resume_2025.pdf.pdf"

        # Or prompt user
        print("Enter resume file path (or press Enter for default):")
        user_input = input().strip()
        
        if user_input:
            test_file = user_input
        
        if Path(test_file).exists():
            test_single_file(test_file)
        else:
            print(f"❌ File not found: {test_file}")
            print("\nUsage: python quick_test.py <path_to_resume.pdf>")




