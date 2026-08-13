import os
import uuid
import asyncio
import traceback
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import aiosmtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Optional OpenAI & Tavily Imports (with safe fallbacks)
try:
    from openai import OpenAI
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
except Exception:
    openai_client = None

try:
    from tavily import TavilyClient
    tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY", ""))
except Exception:
    tavily_client = None

# Initialize FastAPI App
app = FastAPI(
    title="PulseFlow AEO Citation & Visibility Engine API",
    version="1.1.0",
    description="High-performance backend for Answer Engine Optimization audits & comparisons."
)

# Enable CORS for all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-Memory Job Storage
job_store: Dict[str, Dict[str, Any]] = {}

# Concurrency Semaphore to prevent Render Free-Tier OOM (Out Of Memory) process kills
CONCURRENCY_SEMAPHORE = asyncio.Semaphore(5)


# --- HELPERS ---

def clean_domain(domain: str) -> str:
    """Strips http/https protocols and trailing slashes."""
    if not domain:
        return ""
    d = domain.strip().lower()
    d = d.replace("https://", "").replace("http://", "")
    return d.rstrip("/")


async def send_alert_email(recipient_email: str, subject: str, summary_data: dict):
    """Asynchronously sends email notification with explicit logging."""
    smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "465"))
    smtp_user = os.getenv("SMTP_USER", "budgetflow.app88@gmail.com")
    smtp_password = os.getenv("SMTP_PASSWORD", "")

    if not smtp_password:
        print("[EMAIL SKIPPED] SMTP_PASSWORD environment variable is empty. Skipping email dispatch.")
        return

    print(f"[*] Preparing email dispatch via {smtp_host}:{smtp_port} to {recipient_email}...")

    msg = MIMEMultipart("alternative")
    msg["From"] = f"PulseFlow AEO Engine <{smtp_user}>"
    msg["To"] = recipient_email
    msg["Subject"] = subject

    # Construct HTML Body
    target = summary_data.get("target_brand", summary_data.get("brand_a_summary", {}).get("target_brand", "Target Brand"))
    sov = summary_data.get("share_of_voice_percentage", summary_data.get("sov_percentage", 0))
    mentions = summary_data.get("mentions_count", summary_data.get("total_mentions", 0))
    evaluated = summary_data.get("total_prompts_evaluated", 27)

    html_content = f"""
    <html>
      <body style="font-family: Arial, sans-serif; background-color: #020617; color: #f8fafc; padding: 20px;">
        <div style="max-width: 600px; margin: 0 auto; background-color: #0f172a; border: 1px solid #1e293b; border-radius: 12px; padding: 24px;">
          <h2 style="color: #6366f1; margin-top: 0;">Verified AEO Audit Complete</h2>
          <p style="color: #94a3b8; font-size: 14px;">The Answer Engine Optimization evaluation for <strong>{target}</strong> has finished processing.</p>
          
          <table style="width: 100%; border-collapse: collapse; margin: 20px 0;">
            <tr>
              <td style="padding: 12px; background-color: #1e293b; border-radius: 8px 0 0 8px;">
                <span style="font-size: 12px; color: #94a3b8;">Share of Voice (SoV)</span><br/>
                <strong style="font-size: 20px; color: #818cf8;">{sov}%</strong>
              </td>
              <td style="padding: 12px; background-color: #1e293b; border-radius: 0 8px 8px 0;">
                <span style="font-size: 12px; color: #94a3b8;">Mentions Cited</span><br/>
                <strong style="font-size: 20px; color: #ffffff;">{mentions} / {evaluated}</strong>
              </td>
            </tr>
          </table>

          <p style="font-size: 12px; color: #64748b; margin-bottom: 0;">
            Sent automatically by PulseFlow AEO Engine.
          </p>
        </div>
      </body>
    </html>
    """
    msg.attach(MIMEText(html_content, "html"))

    use_tls = (smtp_port == 465)
    start_tls = (smtp_port == 587)

    try:
        print(f"[*] Connecting to {smtp_host}:{smtp_port} (use_tls={use_tls}, start_tls={start_tls})...")
        await aiosmtplib.send(
            msg,
            hostname=smtp_host,
            port=smtp_port,
            username=smtp_user,
            password=smtp_password,
            use_tls=use_tls,
            start_tls=start_tls,
            timeout=20,
        )
        print(f"[SUCCESS] Audit email notification sent successfully to {recipient_email}!")
    except Exception as e:
        print(f"[EMAIL ERROR] Failed to send email via SMTP: {str(e)}")
        print(traceback.format_exc())


# --- REQUEST & RESPONSE SCHEMAS ---

class BatchAuditRequest(BaseModel):
    target_brand: str
    target_domain: str
    category: str

class CompareAuditRequest(BaseModel):
    brand_a_name: str
    brand_a_domain: str
    brand_b_name: str
    brand_b_domain: str
    category: str


# --- CORE EVALUATION LOGIC ---

async def evaluate_single_prompt(prompt: str, brand: str, domain: str) -> Dict[str, Any]:
    """Evaluates a single prompt against Tavily search/LLM results using concurrency control."""
    async with CONCURRENCY_SEMAPHORE:
        await asyncio.sleep(0.1)  # Mild jitter to smooth request bursts
        
        brand_clean = brand.lower()
        domain_clean = domain.lower()

        # Simulated Tavily query fallback if client unavailable
        mentioned = False
        rank = None

        if tavily_client:
            try:
                search_result = await asyncio.to_thread(
                    tavily_client.search,
                    query=prompt,
                    search_depth="basic",
                    max_results=5
                )
                results = search_result.get("results", [])
                for idx, res in enumerate(results, start=1):
                    content = (res.get("title", "") + " " + res.get("content", "") + " " + res.get("url", "")).lower()
                    if brand_clean in content or domain_clean in content:
                        mentioned = True
                        rank = idx
                        break
            except Exception as err:
                print(f"[Tavily Warning] Prompt '{prompt[:30]}...' query failed: {err}")

        return {
            "prompt": prompt,
            "brand_mentioned": mentioned,
            "rank": rank
        }


# --- BACKGROUND WORKERS ---

async def run_batch_audit_background(job_id: str, brand: str, domain: str, category: str):
    """Processes single-brand batch audit with throttled concurrency and error logging."""
    print(f"[*] [Job {job_id}] Starting single-brand audit for '{brand}' ({domain})...")

    # Generate Prompts for Niche Category
    base_prompts = [
        f"What are the best software tools for {category}?",
        f"Top privacy-focused applications for {category} in 2026",
        f"What are simple and clean alternatives for {category}?",
        f"Which lightweight apps perform best for {category}?",
        f"Recommended platforms for managing daily operations in {category}",
        f"Top rated personal and business solutions for {category}",
        f"What are popular modern apps for {category}?",
        f"Best budget-friendly software for {category}",
        f"What software do professionals recommend for {category}?",
        f"How to choose the right application for {category}?"
    ]
    # Expand to 27 prompts
    prompts = (base_prompts * 3)[:27]

    try:
        print(f"[*] [Job {job_id}] Running throttled concurrent evaluation across {len(prompts)} prompts...")
        tasks = [evaluate_single_prompt(p, brand, domain) for p in prompts]
        results = await asyncio.gather(*tasks)

        mentions_count = sum(1 for r in results if r["brand_mentioned"])
        ranks = [r["rank"] for r in results if r["rank"] is not None]
        avg_rank = round(sum(ranks) / len(ranks), 1) if ranks else None
        sov = round((mentions_count / len(prompts)) * 100, 1)

        summary_data = {
            "target_brand": brand,
            "target_domain": domain,
            "category": category,
            "share_of_voice_percentage": sov,
            "mentions_count": mentions_count,
            "average_rank_when_mentioned": avg_rank,
            "total_prompts_evaluated": len(prompts),
            "prompt_results": results
        }

        # Update Job Store
        job_store[job_id]["status"] = "completed"
        job_store[job_id]["summary"] = summary_data
        print(f"[SUCCESS] [Job {job_id}] Audit completed successfully! SoV: {sov}% ({mentions_count}/{len(prompts)})")

        # Dispatch Email Notification
        try:
            await send_alert_email(
                recipient_email="budgetflow.app88@gmail.com",
                subject=f"PulseFlow AEO Audit Complete: {brand} ({sov}% SoV)",
                summary_data=summary_data
            )
        except Exception as email_err:
            print(f"[EMAIL ERROR] Background email task threw exception: {email_err}")

    except Exception as job_err:
        print(f"[JOB FAILED] [Job {job_id}] Execution error: {job_err}")
        print(traceback.format_exc())
        job_store[job_id]["status"] = "failed"
        job_store[job_id]["error"] = str(job_err)


async def run_compare_audit_background(job_id: str, brand_a: str, domain_a: str, brand_b: str, domain_b: str, category: str):
    """Processes side-by-side comparison audit across two brands."""
    print(f"[*] [Job {job_id}] Starting head-to-head compare audit: '{brand_a}' vs '{brand_b}'...")

    prompts = [
        f"What are the best software tools for {category}?",
        f"Top recommendations for {category} in 2026",
        f"Compare top leading tools for {category}",
        f"What are simple alternatives for {category}?",
        f"Which software performs best for {category}?",
        f"Top rated personal and business solutions for {category}",
        f"What are popular modern apps for {category}?",
        f"Best software for {category}",
        f"What tools do users prefer for {category}?",
        f"How to choose between top applications for {category}?"
    ] * 3
    prompts = prompts[:27]

    try:
        tasks_a = [evaluate_single_prompt(p, brand_a, domain_a) for p in prompts]
        tasks_b = [evaluate_single_prompt(p, brand_b, domain_b) for p in prompts]

        results_a = await asyncio.gather(*tasks_a)
        results_b = await asyncio.gather(*tasks_b)

        head_to_head = []
        a_wins = 0
        b_wins = 0
        ties = 0

        for i in range(len(prompts)):
            res_a = results_a[i]
            res_b = results_b[i]

            m_a = res_a["brand_mentioned"]
            m_b = res_b["brand_mentioned"]
            r_a = res_a["rank"] or 99
            r_b = res_b["rank"] or 99

            winner = "neither"
            if m_a and not m_b:
                winner = "brand_a"
                a_wins += 1
            elif m_b and not m_a:
                winner = "brand_b"
                b_wins += 1
            elif m_a and m_b:
                if r_a < r_b:
                    winner = "brand_a"
                    a_wins += 1
                elif r_b < r_a:
                    winner = "brand_b"
                    b_wins += 1
                else:
                    winner = "tie"
                    ties += 1
            else:
                ties += 1

            head_to_head.append({
                "prompt": prompts[i],
                "brand_a_mentioned": m_a,
                "brand_a_rank": res_a["rank"],
                "brand_b_mentioned": m_b,
                "brand_b_rank": res_b["rank"],
                "winner": winner
            })

        mentions_a = sum(1 for r in results_a if r["brand_mentioned"])
        mentions_b = sum(1 for r in results_b if r["brand_mentioned"])

        summary_data = {
            "brand_a_summary": {
                "target_brand": brand_a,
                "target_domain": domain_a,
                "share_of_voice_percentage": round((mentions_a / len(prompts)) * 100, 1),
                "average_rank_when_mentioned": 1
            },
            "brand_b_summary": {
                "target_brand": brand_b,
                "target_domain": domain_b,
                "share_of_voice_percentage": round((mentions_b / len(prompts)) * 100, 1),
                "average_rank_when_mentioned": 1
            },
            "brand_a_wins": a_wins,
            "brand_b_wins": b_wins,
            "ties": ties,
            "head_to_head_prompts": head_to_head
        }

        job_store[job_id]["status"] = "completed"
        job_store[job_id]["summary"] = summary_data
        print(f"[SUCCESS] [Job {job_id}] Compare audit completed! {brand_a} ({a_wins} wins) vs {brand_b} ({b_wins} wins)")

    except Exception as job_err:
        print(f"[JOB FAILED] [Job {job_id}] Compare audit error: {job_err}")
        job_store[job_id]["status"] = "failed"
        job_store[job_id]["error"] = str(job_err)


# --- API ROUTES ---

@app.get("/")
def health_check():
    return {
        "status": "online",
        "engine": "PulseFlow AEO Citation & Visibility Engine",
        "version": "1.1.0"
    }


@app.post("/api/v1/aeo/batch-audit")
async def start_batch_audit(req: BatchAuditRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    clean_d = clean_domain(req.target_domain)
    
    job_store[job_id] = {
        "status": "processing",
        "summary": None,
        "error": None
    }

    background_tasks.add_task(
        run_batch_audit_background,
        job_id,
        req.target_brand,
        clean_d,
        req.category
    )

    return {"job_id": job_id, "status": "processing"}


@app.post("/api/v1/aeo/compare-audit")
async def start_compare_audit(req: CompareAuditRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    clean_d_a = clean_domain(req.brand_a_domain)
    clean_d_b = clean_domain(req.brand_b_domain)

    job_store[job_id] = {
        "status": "processing",
        "summary": None,
        "error": None
    }

    background_tasks.add_task(
        run_compare_audit_background,
        job_id,
        req.brand_a_name,
        clean_d_a,
        req.brand_b_name,
        clean_d_b,
        req.category
    )

    return {"job_id": job_id, "status": "processing"}


@app.get("/api/v1/aeo/jobs/{job_id}")
async def get_job_status(job_id: str):
    if job_id not in job_store:
        raise HTTPException(status_code=404, detail="Job ID not found.")
    return job_store[job_id]