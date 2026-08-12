import uuid
import json
from pathlib import Path
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from aeo_batch_processor import AEOBatchRunner, BatchAuditSummary
from aeo_notifier import AEONotifier

JOBS_DIR = Path(__file__).parent / "jobs"
SCHEDULES_FILE = Path(__file__).parent / "scheduled_monitors.json"

class AEOMonitorScheduler:
    def __init__(self):
        self.scheduler = AsyncIOScheduler()
        self.notifier = AEONotifier()

    def start(self):
        if not self.scheduler.running:
            self.scheduler.start()
            print("[*] AEO Background Scheduler service started.")

    async def execute_scheduled_audit(
        self, 
        user_email: str, 
        target_brand: str, 
        target_domain: str, 
        category: str, 
        drop_threshold: float = 5.0
    ):
        """Runs periodic audit, compares against previous run, and alerts if SoV drops."""
        print(f"[*] Executing scheduled audit for brand: {target_brand}...")
        
        # 1. Locate latest previous audit for this brand from disk
        previous_sov = None
        for job_file in sorted(JOBS_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True):
            try:
                data = json.loads(job_file.read_text())
                summary = data.get("summary", {})
                if summary.get("target_brand") == target_brand and "share_of_voice_percentage" in summary:
                    previous_sov = summary["share_of_voice_percentage"]
                    break
            except Exception:
                continue

        # 2. Execute new audit
        runner = AEOBatchRunner(max_concurrent_requests=5)
        new_summary: BatchAuditSummary = await runner.run_batch_audit(
            target_brand=target_brand,
            target_domain=target_domain,
            category=category
        )

        # 3. Save new job to disk
        job_id = str(uuid.uuid4())
        job_file = JOBS_DIR / f"{job_id}.json"
        job_file.write_text(json.dumps({
            "job_id": job_id,
            "status": "completed",
            "summary": new_summary.model_dump()
        }, indent=2))

        current_sov = new_summary.share_of_voice_percentage

        # 4. Compare SoV and send email alert if drop exceeds threshold
        if previous_sov is not None:
            drop = previous_sov - current_sov
            print(f"[*] Brand: {target_brand} | Previous SoV: {previous_sov}% | Current SoV: {current_sov}% | Drop: {drop}%")

            if drop >= drop_threshold:
                # Find prompts where brand lost placement
                lost_prompts = [
                    r.prompt_evaluated for r in new_summary.individual_reports 
                    if not r.target_brand_mentioned
                ]

                report_url = f"http://localhost:3000/report/{job_id}"
                
                await self.notifier.send_sov_drop_alert(
                    user_email=user_email,
                    target_brand=target_brand,
                    previous_sov=previous_sov,
                    current_sov=current_sov,
                    lost_prompts=lost_prompts,
                    report_url=report_url
                )

    def add_monitoring_job(
        self,
        user_email: str,
        target_brand: str,
        target_domain: str,
        category: str,
        interval_hours: int = 24,
        drop_threshold: float = 5.0
    ):
        """Schedules a recurring background task."""
        job_id = f"monitor_{target_brand.lower().replace(' ', '_')}"
        
        self.scheduler.add_job(
            self.execute_scheduled_audit,
            trigger="interval",
            hours=interval_hours,
            id=job_id,
            replace_existing=True,
            kwargs={
                "user_email": user_email,
                "target_brand": target_brand,
                "target_domain": target_domain,
                "category": category,
                "drop_threshold": drop_threshold
            }
        )
        print(f"[+] Scheduled monitoring added for {target_brand} (Every {interval_hours} hours).")