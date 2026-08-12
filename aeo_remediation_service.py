import os
import json
import asyncio  # Added top-level import
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Import schema structures from your existing setup
from aeo_batch_processor import BatchAuditSummary, AEOExtractionReport

load_dotenv()

# -------------------------------------------------------------------
# 1. PYDANTIC SCHEMAS FOR REMEDIATION GENERATION
# -------------------------------------------------------------------

class SchemaMarkupOutput(BaseModel):
    json_ld_schema: str = Field(
        ..., 
        description="A valid, raw stringified JSON-LD script block containing Organization/SoftwareApplication and FAQPage markup."
    )
    implementation_guide: str = Field(
        ..., 
        description="Clear instructions on where to paste this <script> tag on the domain."
    )

class OutreachEmail(BaseModel):
    target_url: str = Field(..., description="The listicle URL where competitors are cited.")
    recipient_role: str = Field(..., description="Target persona (e.g., 'Editor in Chief' or 'SEO Content Manager').")
    subject_line: str = Field(..., description="Catchy, high-open subject line.")
    email_body: str = Field(..., description="Personalized outreach pitch explaining why the brand should be added to the listicle.")

class OutreachBatchOutput(BaseModel):
    outreach_pitches: List[OutreachEmail] = Field(
        ..., 
        description="Generated cold outreach emails targeting top unlinked listicles."
    )

class AEORemediationPackage(BaseModel):
    target_brand: str
    target_domain: str
    schema_markup: SchemaMarkupOutput
    outreach_campaigns: OutreachBatchOutput

# -------------------------------------------------------------------
# 2. REMEDIATION GENERATOR ENGINE
# -------------------------------------------------------------------

class AEORemediationGenerator:
    def __init__(self):
        raw_key = os.getenv("OPENAI_API_KEY", "")
        api_key = raw_key.strip().strip("'").strip('"')
        if not api_key:
            raise ValueError("OPENAI_API_KEY is missing!")
        self.openai_client = AsyncOpenAI(api_key=api_key)

    async def generate_json_ld_schema(
        self, target_brand: str, target_domain: str, category: str, audit_summary: BatchAuditSummary
    ) -> SchemaMarkupOutput:
        """Generates tailored JSON-LD schema (SoftwareApplication & FAQPage) to fix citation gaps."""
        
        # Extract common questions from prompts where target brand was missing
        missing_prompts = [
            r.prompt_evaluated for r in audit_summary.individual_reports 
            if not r.target_brand_mentioned
        ][:5]

        system_instruction = (
            "You are a Senior Technical SEO and AEO Specialist. Generate clean, valid JSON-LD schema markup "
            "designed to maximize entity understanding for LLM scrapers (ChatGPT, Perplexity, Claude). "
            "Combine `@type`: `SoftwareApplication` (or `Organization`) with an `@type`: `FAQPage` "
            "targeting missing keyword questions."
        )

        user_content = f"""
        TARGET BRAND: {target_brand}
        TARGET DOMAIN: {target_domain}
        CATEGORY: {category}
        
        KEY QUESTIONS AI ENGINE IS MISSING US ON:
        {json.dumps(missing_prompts, indent=2)}
        
        Output valid stringified JSON-LD including:
        1. SoftwareApplication schema with name, operatingSystem, applicationCategory, and offers (Free / Paid).
        2. FAQPage schema addressing the missing prompt questions.
        3. sameAs array referencing standard entity pages (Wikidata, Crunchbase, G2, Trustpilot).
        """

        response = await self.openai_client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_content}
            ],
            response_format=SchemaMarkupOutput,
            temperature=0.2
        )

        return response.choices[0].message.parsed

    async def generate_outreach_emails(
        self, target_brand: str, target_domain: str, audit_summary: BatchAuditSummary
    ) -> OutreachBatchOutput:
        """Finds listicle URLs that cited competitors but missed the target brand and drafts pitch emails."""
        
        # Identify listicle citations where competitor = true and target = false
        unlinked_listicles = []
        for report in audit_summary.individual_reports:
            for cite in report.citations:
                if cite.supports_competitor and not cite.supports_target and cite.source_type == "blog_listicle":
                    if cite.url not in unlinked_listicles:
                        unlinked_listicles.append(cite.url)

        target_urls = unlinked_listicles[:4]  # Focus on top 4 listicles
        if not target_urls:
            # Fallback if all sources are already supporting or non-listicle
            target_urls = audit_summary.top_citation_urls[:3]

        system_instruction = (
            "You are an expert Content Partnership and PR Strategist. "
            "Draft high-converting, non-spammy cold outreach emails to blog editors requesting inclusion "
            "of our brand into their existing roundups/listicles."
        )

        user_content = f"""
        TARGET BRAND: {target_brand}
        TARGET DOMAIN: {target_domain}
        
        LISTICLE URLS TO TARGET FOR INCLUSION:
        {json.dumps(target_urls, indent=2)}
        
        TOP COMPETITORS CURRENTLY FEATURED:
        {json.dumps(list(audit_summary.competitor_mentions_summary.keys())[:5], indent=2)}
        
        For each URL, write a personalized, concise email body pitch highlighting why adding {target_brand} 
        improves their article for readers (e.g. unique feature, free tier, or recent 2026 update).
        """

        response = await self.openai_client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_content}
            ],
            response_format=OutreachBatchOutput,
            temperature=0.4
        )

        return response.choices[0].message.parsed

    async def build_full_package(self, audit_summary: BatchAuditSummary) -> AEORemediationPackage:
        """Executes full remediation pipeline."""
        schema_task = self.generate_json_ld_schema(
            audit_summary.target_brand,
            audit_summary.target_domain,
            audit_summary.category,
            audit_summary
        )
        outreach_task = self.generate_outreach_emails(
            audit_summary.target_brand,
            audit_summary.target_domain,
            audit_summary
        )

        # Run concurrently
        schema_out, outreach_out = await asyncio.gather(schema_task, outreach_task)

        return AEORemediationPackage(
            target_brand=audit_summary.target_brand,
            target_domain=audit_summary.target_domain,
            schema_markup=schema_out,
            outreach_campaigns=outreach_out
        )

# -------------------------------------------------------------------
# 3. STANDALONE TESTING RUNNER
# -------------------------------------------------------------------

if __name__ == "__main__":
    from pathlib import Path

    async def test_remediation():
        print("Testing Remediation Generator with local JSON report...")
        
        # Load most recent job file from jobs/ directory
        jobs_dir = Path(__file__).parent / "jobs"
        job_files = list(jobs_dir.glob("*.json"))
        
        if not job_files:
            print("No job files found in jobs/. Run a batch audit first!")
            return

        latest_job_path = max(job_files, key=os.path.getmtime)
        print(f"Loading audit file: {latest_job_path.name}")
        
        raw_data = json.loads(latest_job_path.read_text())
        summary_dict = raw_data.get("summary")
        
        if not summary_dict:
            print("Job file contains no summary payload.")
            return

        summary = BatchAuditSummary(**summary_dict)
        generator = AEORemediationGenerator()
        
        package = await generator.build_full_package(summary)

        print("\n================ GENERATED JSON-LD SCHEMA ================")
        print(package.schema_markup.json_ld_schema)
        print("\nImplementation Guide:", package.schema_markup.implementation_guide)
        
        print("\n================ GENERATED OUTREACH CAMPAIGNS ================")
        for idx, pitch in enumerate(package.outreach_campaigns.outreach_pitches, 1):
            print(f"\n--- Pitch #{idx} ---")
            print(f"Target URL: {pitch.target_url}")
            print(f"Subject: {pitch.subject_line}")
            print(f"Body:\n{pitch.email_body}\n")

    asyncio.run(test_remediation())