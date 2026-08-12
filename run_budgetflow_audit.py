import asyncio
import httpx
import json

API_BASE_URL = "http://localhost:8000/api/v1/aeo"

async def execute_budgetflow_audit():
    print("==================================================")
    print("   BUDGETFLOW AEO AUDIT & DEMO RUNNER            ")
    print("==================================================")
    print("[*] Initiating 30-Prompt AEO Audit for BudgetFlow...")
    
    payload = {
        "target_brand": "BudgetFlow",
        "target_domain": "budgetflow.app",
        "category": "Personal Finance & Multi-Currency Budgeting"
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        # 1. Trigger Batch Job
        res = await client.post(f"{API_BASE_URL}/batch-audit", json=payload)
        if res.status_code != 200:
            print(f"[Error] Failed to start job: {res.text}")
            return

        data = res.json()
        job_id = data["job_id"]
        print(f"[+] Audit Job Created! Job ID: {job_id}")
        print("[*] Evaluating 30 prompts concurrently across AI Answer Engines...\n")

        # 2. Poll for Completion
        while True:
            await asyncio.sleep(3)
            status_res = await client.get(f"{API_BASE_URL}/jobs/{job_id}")
            if status_res.status_code != 200:
                continue

            job_data = status_res.json()
            status = job_data.get("status")

            if status == "completed":
                summary = job_data["summary"]
                print("==================================================")
                print("              AUDIT COMPLETED RESULTS             ")
                print("==================================================")
                print(f" Target Brand:           {summary['target_brand']}")
                print(f" Share of Voice (SoV):   {summary['share_of_voice_percentage']}%")
                print(f" Avg Rank Position:      #{summary['average_rank_when_mentioned']}")
                print(f" Prompts Evaluated:      {summary['total_prompts_evaluated']}")
                
                competitors = list(summary['competitor_mentions_summary'].keys())[:5]
                print(f" Top Rivals Mentioned:   {', '.join(competitors)}")
                print("--------------------------------------------------")
                print(f"🔗 View Visual Dashboard: http://localhost:3000/report/{job_id}")
                print("==================================================\n")
                break
            elif status == "failed":
                print(f"[Error] Audit failed: {job_data.get('error')}")
                break

if __name__ == "__main__":
    asyncio.run(execute_budgetflow_audit())