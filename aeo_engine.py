import os
import uuid
import json
import asyncio
import traceback
from pathlib import Path
from typing import List, Optional, Dict, Literal, Any
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import httpx

# Automatically load variables from .env file into os.environ
load_dotenv()

# Create 'jobs' directory to store reports persistently on disk
JOBS_DIR = Path(__file__).parent / "jobs"
JOBS_DIR.mkdir(exist_ok=True)

# -------------------------------------------------------------------
# 1. PYDANTIC SCHEMAS
# -------------------------------------------------------------------

class CompetitorMention(BaseModel):
    name: str = Field(..., description="Name of the competing brand recommended by the AI.")
    rank: int = Field(..., description="Position in which the competitor was listed (1-based).")
    key_reason: str = Field(..., description="Why the AI recommended them (e.g. 'Cheapest option').")

class CitationSource(BaseModel):
    url: str = Field(..., description="The URL cited or referenced by the Answer Engine.")
    source_type: Literal["reddit", "g2_capterra", "blog_listicle", "news", "official_site", "other"] = Field(
        ..., description="Category of source domain."
    )
    supports_competitor: bool = Field(..., description="True if this page promotes a competitor.")
    supports_target: bool = Field(..., description="True if this page promotes our target brand.")

class AEOExtractionReport(BaseModel):
    target_brand: str = Field(..., description="The user's brand being analyzed.")
    prompt_evaluated: str = Field(..., description="The exact prompt queried into the Answer Engine.")
    target_brand_mentioned: bool = Field(..., description="Whether the target brand appears in the response.")
    target_brand_rank: Optional[int] = Field(None, description="Rank position of target brand if mentioned.")
    sentiment: Literal["positive", "neutral", "negative", "absent"] = Field(..., description="Brand positioning sentiment.")
    competitors_mentioned: List[CompetitorMention] = Field(default_factory=list)
    citations: List[CitationSource] = Field(default_factory=list)
    remediation_actions: List[str] = Field(
        ..., min_length=3, max_length=5, description="Specific actionable steps to win this prompt position."
    )

class AEORequest(BaseModel):
    target_brand: str
    target_domain: str
    prompt: str

class BatchAuditRequest(BaseModel):
    target_brand: str
    target_domain: str
    category: str
    custom_prompts: Optional[List[str]] = None

class CompareAuditRequest(BaseModel):
    brand_a_name: str
    brand_a_domain: str
    brand_b_name: str
    brand_b_domain: str
    category: str

class HeadToHeadPromptResult(BaseModel):
    prompt: str
    brand_a_mentioned: bool
    brand_a_rank: Optional[int] = None
    brand_b_mentioned: bool
    brand_b_rank: Optional[int] = None
    winner: Literal["brand_a", "brand_b", "tie", "neither"]

class CompareAuditSummary(BaseModel):
    category: str
    total_prompts: int
    brand_a_summary: Any
    brand_b_summary: Any
    head_to_head_prompts: List[HeadToHeadPromptResult]
    brand_a_wins: int
    brand_b_wins: int
    ties: int

class JobStatusResponse(BaseModel):
    job_id: str
    status: Literal["processing", "completed", "failed"]
    summary: Optional[Any] = None
    error: Optional[str] = None

# In-memory database cache
JOBS_DB: Dict[str, JobStatusResponse] = {}

# -------------------------------------------------------------------
# 2. PIPELINE CORE ENGINE
# -------------------------------------------------------------------

class AEOEngine:
    def __init__(self):
        raw_key = os.getenv("OPENAI_API_KEY", "")
        api_key = raw_key.strip().strip("'").strip('"')
        
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is missing! Ensure your .env file contains OPENAI_API_KEY=your_key_here"
            )
        self.openai_client = AsyncOpenAI(api_key=api_key)
        
        raw_tavily = os.getenv("TAVILY_API_KEY", "")
        self.tavily_api_key = raw_tavily.strip().strip("'").strip('"')

    async def _fetch_web_grounded_answer(self, prompt: str) -> dict:
        """Simulates an Answer Engine query via Tavily Search API (or fallback)."""
        if not self.tavily_api_key:
            return {
                "answer": f"When looking for {prompt}, top tools include CompetitorX and CompetitorY. CompetitorX is praised on Reddit for affordability.",
                "urls": ["https://reddit.com/r/SaaS/comments/best_tools", "https://g2.com/categories/software"]
            }
            
        async with httpx.AsyncClient(timeout=15.0) as client:
            res = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": self.tavily_api_key, 
                    "query": prompt, 
                    "search_depth": "advanced", 
                    "include_answer": True
                }
            )
            if res.status_code != 200:
                print(f"[Warning] Tavily search returned status {res.status_code}: {res.text}")
                return {
                    "answer": f"Search failed with status {res.status_code}. Unable to retrieve live web results.",
                    "urls": []
                }
            data = res.json()
            answer = data.get("answer") or "No direct summary generated by search provider."
            urls = [result["url"] for result in data.get("results", []) if "url" in result]
            return {"answer": answer, "urls": urls}

    async def analyze_prompt(self, target_brand: str, target_domain: str, prompt: str) -> AEOExtractionReport:
        web_data = await self._fetch_web_grounded_answer(prompt)
        raw_answer = web_data["answer"]
        cited_urls = web_data["urls"]

        system_instruction = (
            "You are an expert AEO (Answer Engine Optimization) Analyst. "
            "Examine the raw answer generated by an AI Answer Engine and its cited URLs. "
            "Extract brand mentions, evaluate target brand presence, parse citations, "
            "and output 3 high-value remediation steps to capture this recommendation spot."
        )

        user_content = f"""
        TARGET BRAND: {target_brand} ({target_domain})
        PROMPT QUERY: {prompt}
        
        RAW ANSWER ENGINE OUTPUT:
        {raw_answer}
        
        CITATIONS / SOURCE URLS RETRIEVED:
        {cited_urls}
        """

        response = await self.openai_client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_content}
            ],
            response_format=AEOExtractionReport,
            temperature=0.1
        )

        return response.choices[0].message.parsed

# -------------------------------------------------------------------
# 3. BACKGROUND TASK WORKERS
# -------------------------------------------------------------------

async def process_batch_job(job_id: str, req: BatchAuditRequest):
    """Worker function executed asynchronously in background for single brand audit."""
    try:
        from aeo_batch_processor import AEOBatchRunner
        
        runner = AEOBatchRunner(max_concurrent_requests=5)
        summary = await runner.run_batch_audit(
            target_brand=req.target_brand,
            target_domain=req.target_domain,
            category=req.category,
            custom_prompts=req.custom_prompts
        )
        
        job_data = JobStatusResponse(
            job_id=job_id,
            status="completed",
            summary=summary
        )
        
        JOBS_DB[job_id] = job_data
        job_file = JOBS_DIR / f"{job_id}.json"
        job_file.write_text(job_data.model_dump_json(indent=2))
        
    except Exception as e:
        print(f"[Error] Background Job {job_id} failed: {e}")
        traceback.print_exc()
        
        failed_job = JobStatusResponse(
            job_id=job_id,
            status="failed",
            error=str(e)
        )
        JOBS_DB[job_id] = failed_job
        (JOBS_DIR / f"{job_id}.json").write_text(failed_job.model_dump_json(indent=2))


async def process_compare_job(job_id: str, req: CompareAuditRequest):
    """Worker function executing dual 30-prompt head-to-head audits concurrently."""
    try:
        from aeo_batch_processor import AEOBatchRunner
        
        runner = AEOBatchRunner(max_concurrent_requests=5)
        print(f"[*] Generating shared 30-prompt taxonomy for category: '{req.category}'...")
        shared_prompts = await runner.prompt_generator.generate_30_prompts(
            target_brand=req.brand_a_name, category=req.category
        )

        print(f"[*] Executing dual concurrent audits: {req.brand_a_name} vs {req.brand_b_name}...")
        task_a = runner.run_batch_audit(
            target_brand=req.brand_a_name,
            target_domain=req.brand_a_domain,
            category=req.category,
            custom_prompts=shared_prompts
        )
        task_b = runner.run_batch_audit(
            target_brand=req.brand_b_name,
            target_domain=req.brand_b_domain,
            category=req.category,
            custom_prompts=shared_prompts
        )

        summary_a, summary_b = await asyncio.gather(task_a, task_b)

        head_to_head: List[HeadToHeadPromptResult] = []
        wins_a = 0
        wins_b = 0
        ties = 0

        for r_a, r_b in zip(summary_a.individual_reports, summary_b.individual_reports):
            p_text = r_a.prompt_evaluated
            a_m, a_r = r_a.target_brand_mentioned, r_a.target_brand_rank
            b_m, b_r = r_b.target_brand_mentioned, r_b.target_brand_rank

            if a_m and not b_m:
                winner = "brand_a"
                wins_a += 1
            elif b_m and not a_m:
                winner = "brand_b"
                wins_b += 1
            elif a_m and b_m:
                rank_a = a_r if a_r is not None else 99
                rank_b = b_r if b_r is not None else 99
                if rank_a < rank_b:
                    winner = "brand_a"
                    wins_a += 1
                elif rank_b < rank_a:
                    winner = "brand_b"
                    wins_b += 1
                else:
                    winner = "tie"
                    ties += 1
            else:
                winner = "neither"

            head_to_head.append(HeadToHeadPromptResult(
                prompt=p_text,
                brand_a_mentioned=a_m,
                brand_a_rank=a_r,
                brand_b_mentioned=b_m,
                brand_b_rank=b_r,
                winner=winner
            ))

        comparison_payload = CompareAuditSummary(
            category=req.category,
            total_prompts=len(shared_prompts),
            brand_a_summary=summary_a,
            brand_b_summary=summary_b,
            head_to_head_prompts=head_to_head,
            brand_a_wins=wins_a,
            brand_b_wins=wins_b,
            ties=ties
        )

        job_data = JobStatusResponse(
            job_id=job_id,
            status="completed",
            summary=comparison_payload
        )

        JOBS_DB[job_id] = job_data
        (JOBS_DIR / f"{job_id}.json").write_text(job_data.model_dump_json(indent=2))

    except Exception as e:
        print(f"[Error] Comparison Job {job_id} failed: {e}")
        traceback.print_exc()
        failed_job = JobStatusResponse(job_id=job_id, status="failed", error=str(e))
        JOBS_DB[job_id] = failed_job
        (JOBS_DIR / f"{job_id}.json").write_text(failed_job.model_dump_json(indent=2))

# -------------------------------------------------------------------
# 4. FASTAPI APP & SERVICE ENDPOINTS
# -------------------------------------------------------------------

engine: Optional[AEOEngine] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    engine = AEOEngine()
    print("[*] AEO API Engine initialized successfully.")
    yield

app = FastAPI(
    title="AEO Citation Extraction & Batch Audit API", 
    version="1.0.0", 
    lifespan=lifespan
)

# Enable CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/v1/aeo/audit", response_model=AEOExtractionReport)
async def run_aeo_audit(req: AEORequest):
    """Executes a single prompt audit synchronously."""
    if engine is None:
        raise HTTPException(status_code=500, detail="AEO Engine is not initialized.")
    try:
        report = await engine.analyze_prompt(
            target_brand=req.target_brand,
            target_domain=req.target_domain,
            prompt=req.prompt
        )
        return report
    except Exception as e:
        print("\n================ DETAILED BACKEND ERROR ================")
        traceback.print_exc()
        print("========================================================\n")
        
        error_msg = str(e) or f"{type(e).__name__}: An error occurred during analysis."
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/api/v1/aeo/batch-audit", response_model=Dict[str, str])
async def start_batch_audit(req: BatchAuditRequest, background_tasks: BackgroundTasks):
    """Triggers an asynchronous 30-prompt single brand batch audit."""
    job_id = str(uuid.uuid4())
    
    initial_status = JobStatusResponse(
        job_id=job_id,
        status="processing"
    )
    JOBS_DB[job_id] = initial_status
    background_tasks.add_task(process_batch_job, job_id, req)
    
    return {
        "job_id": job_id,
        "status": "processing",
        "message": "Batch audit started. Poll GET /api/v1/aeo/jobs/{job_id} for completion."
    }

@app.post("/api/v1/aeo/compare-audit", response_model=Dict[str, str])
async def start_compare_audit(req: CompareAuditRequest, background_tasks: BackgroundTasks):
    """Triggers an asynchronous head-to-head comparison audit between two brands."""
    job_id = str(uuid.uuid4())
    
    initial_status = JobStatusResponse(
        job_id=job_id,
        status="processing"
    )
    JOBS_DB[job_id] = initial_status
    background_tasks.add_task(process_compare_job, job_id, req)
    
    return {
        "job_id": job_id,
        "status": "processing",
        "message": "Comparison audit started. Poll GET /api/v1/aeo/jobs/{job_id} for completion."
    }

@app.get("/api/v1/aeo/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Poll status from memory cache or load from JSON file on disk."""
    if job_id in JOBS_DB:
        return JOBS_DB[job_id]
        
    job_file = JOBS_DIR / f"{job_id}.json"
    if job_file.exists():
        data = json.loads(job_file.read_text())
        job_response = JobStatusResponse(**data)
        JOBS_DB[job_id] = job_response
        return job_response

    raise HTTPException(status_code=404, detail="Job ID not found.")

@app.post("/api/v1/aeo/remediation/{job_id}")
async def generate_remediation_package(job_id: str):
    """Generates custom JSON-LD schema and cold outreach emails based on audit gaps."""
    job_status = await get_job_status(job_id)
    
    if job_status.status != "completed" or not job_status.summary:
        raise HTTPException(
            status_code=400, 
            detail="Audit job must be completed before generating remediation package."
        )

    try:
        from aeo_batch_processor import BatchAuditSummary
        from aeo_remediation_service import AEORemediationGenerator
        
        summary_payload = job_status.summary

        # Check if payload is a head-to-head comparison summary and extract brand_a_summary
        if isinstance(summary_payload, dict):
            if "brand_a_summary" in summary_payload:
                summary_obj = BatchAuditSummary(**summary_payload["brand_a_summary"])
            else:
                summary_obj = BatchAuditSummary(**summary_payload)
        elif hasattr(summary_payload, "brand_a_summary"):
            summary_obj = summary_payload.brand_a_summary
        else:
            summary_obj = summary_payload

        generator = AEORemediationGenerator()
        package = await generator.build_full_package(summary_obj)
        return package
    except Exception as e:
        print(f"[Error] Failed to generate remediation package: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("aeo_engine:app", host="0.0.0.0", port=8000, reload=True)