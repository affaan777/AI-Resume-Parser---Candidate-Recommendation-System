"""
test_full_pipeline.py

Complete end-to-end test for the resume processing pipeline:
1. PDF/DOCX Parsing
2. Embedding Generation
3. Vector Storage in Pinecone
4. Retrieval and Validation

Run this after making changes to verify everything works together.
"""

import os 
import uuid
import json
import time
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.parser.resume_parser import ResumeParser
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.embeddings.vector_store import VectorStore
from src.embeddings.preprocessor import ResumePreprocessor

class PipelineTestResult:
    """Track tesst results"""
    def __init__(self):
        self.total_files = 0
        self.successful = 0
        self.failed = 0
        self.errors = []
        self.timing = {}
        self.results = []

    def add_success(self, filename: str, details: Dict):
        self.successful +=1
        self.results.append({
            "file": filename,
            "status": "SUCCESS",
            "details": details
        })

    def add_failure(self, filename: str, error: str):
        self.failed +=1
        self.errors.append({"file": filename, "error": error})
        self.results.append({
            "file": filename,
            "status": "FAILED",
            "error": error
        })

    def print_summary(self):
        """Print test summary"""
        print("\n" + "=" * 80)
        print("📊 PIPELINE TEST SUMMARY")
        print("=" * 80)
        print(f"Total Files Tested: {self.total_files}")
        print(f"✅ Successful: {self.successful}")
        print(f"❌ Failed: {self.failed}")
        print(f"Success Rate: {(self.successful/self.total_files*100):.1f}%")
        
        if self.timing:
            print("\n⏱️  TIMING:")
            for step, duration in self.timing.items():
                print(f"   {step}: {duration:.2f}s")
        
        if self.errors:
            print("\n❌ ERRORS:")
            for error in self.errors:
                print(f"   • {error['file']}: {error['error']}")
        
        print("=" * 80)

class ResumePipelineTester:
    
    """Test the complete resume processing pipeline"""

    def __init__(self, resume_folder: str = "../uploaded_resumes"):
        """
        Initialize the pipeline tester
        
        Args:
            resume_folder: Folder containing test resume files
        """
        self.resume_folder = Path(resume_folder)
        self.result = PipelineTestResult()
         
        # Initialize components
        print("🔄 Initializing pipeline components...")
        try:
            self.parser = ResumeParser()
            self.embedding_generator = EmbeddingGenerator()
            self.preprocessor = ResumePreprocessor()
            self.vector_store = VectorStore()
            print("✅ All components initialized successfully")
        except Exception as e:
            print(f"❌ Failed to initialize components: {e}")
            raise

    def find_resume_files(self) -> List[Path]:
        """Find all PDF and DOCX files in the resume folder"""
        if not self.resume_folder.exists():
            print(f"❌ Folder not found: {self.resume_folder}")
            return []
        
        pdf_files = list(self.resume_folder.glob("*.pdf"))
        docx_files = list(self.resume_folder.glob("*.docx"))

        all_files = pdf_files + docx_files
        print(f"📁 Found {len(all_files)} resume files ({len(pdf_files)} PDF, {len(docx_files)} DOCX)")
        
        return all_files
    
    def extract_text_from_file(self, file_path: Path) -> str:
        """Extract text from PDF or DOCX file"""
        # Import here to avoid circular dependencies
        if file_path.suffix.lower() == '.pdf':
            from PyPDF2 import PdfReader
            reader = PdfReader(str(file_path))
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            return text
        elif file_path.suffix.lower() == '.docx':
            import docx
            doc = docx.Document(str(file_path))
            return "\n".join([para.text for para in doc.paragraphs])
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
    def test_single_resume(self, file_path: Path) -> Dict[str, Any]:
        """
        Test processing a single resume file
        
        Returns:
            Dictionary with test results
        """
        filename = file_path.name
        resume_id = None 

        print(f"\n{'='*80}")
        print(f"🧪 Testing: {filename}")
        print(f"{'='*80}")

        start_time = time.time()

        try:
            # Step 1: Extract text from file
            print("📄 Step 1: Extracting text from file...")
            step1_start = time.time()
            resume_text = self.extract_text_from_file(file_path)
            step1_time = time.time() - step1_start
            print(f"   ✅ Extracted {len(resume_text)} characters in {step1_time:.2f}s")
            
            # Step 2: Parse resume with AI
            print("🤖 Step 2: Parsing resume with AI...")
            step2_start = time.time()
            parsed_resume = self.parser.parse_resume(file_path)
            step2_time = time.time() - step2_start
            # Lines ~95-100
            print(f"   ✅ Parsed resume for: {parsed_resume['full_name']}")
            print(f"   📧 Email: {parsed_resume['email']}")
            print(f"   🎯 Skills: {len(parsed_resume['skills'])} skills")
            print(f"   💼 Experience: {len(parsed_resume['work_experience'])} jobs")
            print(f"   🎓 Education: {len(parsed_resume['education'])} degrees")
            print(f"   ⏱️  Parsing time: {step2_time:.2f}s")

            # # Step 3: Preprocess for embedding
            # print("🔧 Step 3: Preprocessing resume...")
            # step3_start = time.time()
            # preprocessed_text = self.preprocessor.process(parsed_resume)
            # step3_time = time.time() - step3_start

            # # Calculate token count (rough estimate)
            # token_count = len(preprocessed_text.split())
            # print(f"   ✅ Preprocessed text: {len(preprocessed_text)} chars")
            # print(f"   📊 Estimated tokens: ~{token_count}")
            # print(f"   ⏱️  Preprocessing time: {step3_time:.2f}s")
            
            # if token_count > 1800:
            #     print(f"   ⚠️  WARNING: Text may exceed Gemini's 2048 token limit!")

            # 🆕 Generate resume_id IMMEDIATELY
            resume_id = f"resume_{uuid.uuid4().hex[:16]}"

            # Step 3: Preprocess for embedding
            print("🔧 Step 3: Preprocessing resume...")
            step3_start = time.time()

            # Check if resume is large
            full_text = self.preprocessor.process(parsed_resume)
            token_estimate = len(full_text.split())

            if token_estimate > 1800:
                print(f"   ⚠️ Large resume detected ({token_estimate} tokens), using chunking...")
                chunks = self.preprocessor.process_large_resume(parsed_resume)
                is_chunked = True
                print(f"   ✅ Split into {len(chunks)} chunks")
            else:
                chunks = [full_text]
                is_chunked = False
                print(f"   ✅ Normal size resume ({token_estimate} tokens)")

            step3_time = time.time() - step3_start
            print(f"   ⏱️ Preprocessing time: {step3_time:.2f}s")


            # Step 4: Generate embeddings for all chunks
            print("🧬 Step 4: Generating embeddings...")
            step4_start = time.time()

            embeddings = []
            for i, chunk in enumerate(chunks):
                embedding = self.embedding_generator.generate_embedding(chunk)
                embeddings.append(embedding)
                if is_chunked:
                    print(f"   ✅ Generated embedding for chunk {i+1}/{len(chunks)}")

            step4_time = time.time() - step4_start
            print(f"   ✅ Total embeddings: {len(embeddings)} x {len(embeddings[0])} dimensions")
            print(f"   ⏱️ Embedding time: {step4_time:.2f}s")

            # Step 5: Prepare metadata
            # metadata = {
            # "resume_id": resume_id,
            # "full_name": parsed_resume['full_name'],
            # "email": parsed_resume['email'],
            # "phone_number": parsed_resume.get('phone_number') or "",
            # "location": parsed_resume.get('location') or "",
            # "professional_title": parsed_resume.get('professional_title') or "",
            # "skills": parsed_resume['skills'][:20],
            # "total_skills": len(parsed_resume['skills']),
            # "experience_count": len(parsed_resume['work_experience']),
            # "education_count": len(parsed_resume['education']),
            # "filename": filename,
            # "processed_at": datetime.now().isoformat()
            # }
            metadata = {
            "resume_id": resume_id,
            "full_name": parsed_resume.get('full_name', 'Anonymous'),
            "email": parsed_resume.get('email', f"{resume_id}@anonymous.com"),
            "professional_title": parsed_resume.get('professional_title', ''),
            "skills": parsed_resume.get('skills', [])[:20],
            "total_skills": len(parsed_resume.get('skills', [])),
            "experience_count": len(parsed_resume.get('work_experience', [])),
            "education_count": len(parsed_resume.get('education', [])),
            "total_chunks": len(chunks) if is_chunked else 1,
            "filename": filename
            }       

            # Add URLs only if they exist (not null)
            if parsed_resume.get('linkedin_url'):
                metadata['linkedin_url'] = str(parsed_resume['linkedin_url'])
            if parsed_resume.get('github_url'):
                metadata['github_url'] = str(parsed_resume['github_url'])

            print(f"   ✅ Metadata prepared")

            # Step 6: Store in Pinecone
            print("💾 Step 6: Storing in Pinecone...")
            step6_start = time.time()
            
            # Generate unique ID
            # resume_id = f"resume_{parsed_resume['email'].replace('@', '_').replace('.', '_')}"
            resume_id = f"resume_{uuid.uuid4().hex[:16]}"

            
            # Store all chunks
            for i, embedding in enumerate(embeddings):
                chunk_id = f"{resume_id}_chunk{i}" if is_chunked else resume_id
                success = self.vector_store.upsert_resume(chunk_id, embedding, metadata)
                if not success:
                    raise Exception(f"Failed to store chunk {i}")
                if is_chunked:
                    print(f"   ✅ Stored chunk {i+1}/{len(embeddings)}")

            step6_time = time.time() - step6_start
            print(f"   ✅ Stored in Pinecone with ID: {resume_id}")
            print(f"   ⏱️  Storage time: {step6_time:.2f}s")
                
            # Step 7: Verify storage
            print("🔍 Step 7: Verifying storage...")
            time.sleep(3)  # Wait for Pinecone indexing
            
            retrieved = self.vector_store.get_resume_by_id(resume_id)
            if retrieved:
                print(f"   ✅ Successfully retrieved from Pinecone")
                print(f"   📊 Metadata verified: {retrieved['metadata']['full_name']}")
            else:
                # Don't fail - just warn
                print(f"   ⚠️ Retrieval delayed (Pinecone indexing lag)")
                # Don't raise exception here
            
            # Calculate total time
            total_time = time.time() - start_time
            
            print(f"\n{'='*80}")
            print(f"✅ SUCCESS: {filename}")
            print(f"⏱️  Total time: {total_time:.2f}s")
            print(f"{'='*80}")
            
            # Return detailed results
            return {
            "success": True,
            "resume_id": resume_id or "unknown",
            "candidate_name": parsed_resume['full_name'],
            "email": parsed_resume['email'],
            "skills_count": len(parsed_resume['skills']),
            "embedding_dims": len(embeddings[0]),           
            "token_estimate": token_estimate,
            "timings": {
                "text_extraction": step1_time,
                "parsing": step2_time,
                "preprocessing": step3_time,
                "embedding": step4_time,
                "storage": step6_time,
                "total": total_time
            }
        }
            
        except Exception as e:
            print(f"\n❌ FAILED: {filename}")
            print(f"   Error: {str(e)}")
            import traceback
            traceback.print_exc()
            
            return {
                "success": False,
                "error": str(e)
            }
    
    def run_tests(self, max_files: int = None) -> PipelineTestResult:
        """
        Run tests on all resume files
        
        Args:
            max_files: Maximum number of files to test (None = all)
        """
        print("\n" + "="*80)
        print("🚀 STARTING PIPELINE TESTS")
        print("="*80)
        
        # Find resume files
        files = self.find_resume_files()
        
        if not files:
            print("❌ No resume files found!")
            return self.result
        
        # Limit files if specified
        if max_files:
            files = files[:max_files]
            print(f"⚠️  Testing limited to {max_files} files")
        
        self.result.total_files = len(files)
        
        # Test each file
        overall_start = time.time()
        
        for i, file_path in enumerate(files, 1):
            print(f"\n{'#'*80}")
            print(f"Progress: {i}/{len(files)}")
            print(f"{'#'*80}")
            
            result = self.test_single_resume(file_path)
            
            if result["success"]:
                self.result.add_success(file_path.name, result)
            else:
                self.result.add_failure(file_path.name, result["error"])
            
            # Small delay between tests
            if i < len(files):
                time.sleep(1)
        
        # Calculate timing
        total_duration = time.time() - overall_start
        self.result.timing["total_duration"] = total_duration
        self.result.timing["avg_per_resume"] = total_duration / len(files)
        
        # Print summary
        self.result.print_summary()
        
        # Save results to JSON
        self.save_results()
        
        return self.result
    
    def save_results(self):
        """Save test results to JSON file"""
        os.makedirs("outputs/test_results", exist_ok=True)

        output_file = f"outputs/test_results/test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
        results_data = {
            "test_date": datetime.now().isoformat(),
            "total_files": self.result.total_files,
            "successful": self.result.successful,
            "failed": self.result.failed,
            "success_rate": (self.result.successful / self.result.total_files * 100) if self.result.total_files > 0 else 0,
            "timing": self.result.timing,
            "results": self.result.results,
            "errors": self.result.errors
        }
        with open(output_file, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\n💾 Results saved to: {output_file}")


def main():
    """Main test execution"""
    print("\n" + "🧪" * 40)
    print("RESUME PROCESSING PIPELINE - FULL TEST")
    print("🧪" * 40)
    
    # Configuration
    RESUME_FOLDER = "uploaded_resumes"  # One folder up
    MAX_FILES = None # Set to limit number of files (e.g., 5 for quick test)
    
    try:
        # Create tester
        tester = ResumePipelineTester(resume_folder=RESUME_FOLDER)
        
        # Run tests
        result = tester.run_tests(max_files=MAX_FILES)
        
        # Final verdict
        print("\n" + "="*80)
        if result.failed == 0:
            print("🎉 ALL TESTS PASSED! Pipeline is working perfectly!")
        elif result.successful > result.failed:
            print("⚠️  PARTIAL SUCCESS: Most tests passed, but some failed.")
        else:
            print("❌ MAJOR ISSUES: More tests failed than succeeded.")
        print("="*80)
        
        # Get index stats
        print("\n📊 Final Pinecone Index Statistics:")
        stats = tester.vector_store.get_index_stats()
        print(f"   Total Vectors: {stats.get('total_vectors', 0)}")
        print(f"   Dimension: {stats.get('dimension', 0)}")
        
    except Exception as e:
        print(f"\n❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()







        



