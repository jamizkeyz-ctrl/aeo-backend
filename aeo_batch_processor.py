import os
import asyncio
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Import schema and engine components from your existing script
from aeo_engine import AEOEngine, AEOExtractionReport

load_dotenv()

# -------------------------------------------------------------------
# 1. SCHEMAS FOR BATCH PROMPT GENERATION
# -------------------------------------------------------------------

class GeneratedPrompts(BaseModel):
    prompts: List[str] = Field(
        ...,
        min_length=20,
        max_length=30,
        description="List of 20-30 diverse, high-intent buyer prompts across various search intents."
    )

class BatchAuditSummary(BaseModel):
    target_brand: str
    target_domain: str
    category: str
    total_prompts_evaluated: int
    share_of_voice_percentage: float = Field(
        ..., description="Percentage of prompts where target brand was mentioned."
    )
    average_rank_when_mentioned: float = Field(
        ..., description="Average rank position when mentioned (1.0 is best)."
    )
    competitor_mentions_summary: Dict[str, int] = Field(
        ..., description="Total mention count for each competitor across all prompts."
    )
    top_citation_urls: List[str] = Field(
        ..., description="Most frequent citation URLs powering AI answers."
    )
    individual_reports: List[AEOExtractionReport]

# -------------------------------------------------------------------
# 2. PROMPT GENERATOR MODULE
# -------------------------------------------------------------------

class PromptGenerator:
    def __init__(self, openai_client: AsyncOpenAI):
        self.client = openai_client

    async def generate_30_prompts(self, target_brand: str, category: str) -> List[str]:
        """Generates 30 high-intent search prompts across key AEO intent categories."""
        system_instruction = (
            "You are an expert search taxonomy specialist. Generate 30 distinct, natural, high-intent search "
            "prompts that buyers enter into ChatGPT or Answer Engines when researching products in a given niche. "
            "Do NOT include the target brand name in all prompts—mix unbranded discovery queries, comparison "
            "queries, alternative queries, and feature/price-specific queries."
        )

        user_content = f"""
        TARGET BRAND: {target_brand}
        CATEGORY / INDUSTRY: {category}
        
        Generate 30 prompts matching these category proportions:
        - 10 Unbranded Discovery/Best-Of ("best [category] for startups", "top 5 [category] tools")
        - 8 Direct Category Alternatives ("[Category Leader] alternatives", "cheaper options than [Category Leader]")
        - 6 Persona & Feature Specific ("best [category] for small teams", "[category] with free tier")
        - 6 Direct Comparison / Recommendation ("what [category] should I use for X?")
        """

        response = await self.client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_content}
            ],
            response_format=GeneratedPrompts,
            temperature=0.7
        )

        return response.choices[0].message.parsed.prompts

# -------------------------------------------------------------------
# 3. ASYNC BATCH PIPELINE RUNNER
# -------------------------------------------------------------------

class AEOBatchRunner:
    def __init__(self, max_concurrent_requests: int = 5):
        api_key = os.getenv("OPENAI_API_KEY", "").strip().strip("'").strip('"')
        self.openai_client = AsyncOpenAI(api_key=api_key)
        self.engine = AEOEngine()
        self.prompt_generator = PromptGenerator(self.openai_client)
        # Semaphore prevents hitting OpenAI/Tavily rate limits (429 errors)
        self.semaphore = asyncio.Semaphore(max_concurrent_requests)

    async def _audit_single_prompt_safe(
        self, target_brand: str, target_domain: str, prompt: str
    ) -> AEOExtractionReport:
        """Executes single prompt audit bounded by the concurrency semaphore."""
        async with self.semaphore:
            try:
                return await self.engine.analyze_prompt(
                    target_brand=target_brand,
                    target_domain=target_domain,
                    prompt=prompt
                )
            except Exception as e:
                # Return a fallback report if one prompt fails to avoid breaking the entire batch
                print(f"[Warning] Failed prompt execution: '{prompt[:30]}...' -> Error: {e}")
                return AEOExtractionReport(
                    target_brand=target_brand,
                    prompt_evaluated=prompt,
                    target_brand_mentioned=False,
                    target_brand_rank=None,
                    sentiment="absent",
                    competitors_mentioned=[],
                    citations=[],
                    remediation_actions=["Retry audit execution for this specific prompt query."]
                )

    async def run_batch_audit(
        self, target_brand: str, target_domain: str, category: str, custom_prompts: List[str] = None
    ) -> BatchAuditSummary:
        """Generates prompts and runs parallel audit across all queries using asyncio.gather."""
        # Step 1: Generate or assign prompts
        if custom_prompts and len(custom_prompts) > 0:
            prompts = custom_prompts
        else:
            print(f"[*] Generating 30 target prompts for {target_brand} in category '{category}'...")
            prompts = await self.prompt_generator.generate_30_prompts(target_brand, category)

        print(f"[*] Starting concurrent execution across {len(prompts)} prompts...")

        # Step 2: Schedule concurrent tasks with asyncio.gather
        tasks = [
            self._audit_single_prompt_safe(target_brand, target_domain, prompt)
            for prompt in prompts
        ]
        
        # Runs all tasks asynchronously in parallel batches bounded by semaphore
        reports: List[AEOExtractionReport] = await asyncio.gather(*tasks)

        # Step 3: Aggregate SoV and Citations Metrics
        total_prompts = len(reports)
        mentioned_reports = [r for r in reports if r.target_brand_mentioned]
        sov_percentage = (len(mentioned_reports) / total_prompts * 100) if total_prompts > 0 else 0.0

        ranks = [r.target_brand_rank for r in mentioned_reports if r.target_brand_rank is not None]
        avg_rank = (sum(ranks) / len(ranks)) if len(ranks) > 0 else 0.0

        competitor_counts: Dict[str, int] = {}
        citation_counts: Dict[str, int] = {}

        for report in reports:
            for comp in report.competitors_mentioned:
                competitor_counts[comp.name] = competitor_counts.get(comp.name, 0) + 1
            for cite in report.citations:
                citation_counts[cite.url] = citation_counts.get(cite.url, 0) + 1

        top_citations = sorted(citation_counts, key=citation_counts.get, reverse=True)[:10]

        return BatchAuditSummary(
            target_brand=target_brand,
            target_domain=target_domain,
            category=category,
            total_prompts_evaluated=total_prompts,
            share_of_voice_percentage=round(sov_percentage, 2),
            average_rank_when_mentioned=round(avg_rank, 2),
            competitor_mentions_summary=competitor_counts,
            top_citation_urls=top_citations,
            individual_reports=reports
        )

# -------------------------------------------------------------------
# 4. STANDALONE SCRIPT RUNNER (FOR LOCAL TESTING)
# -------------------------------------------------------------------

if __name__ == "__main__":
    async def main():
        batch_runner = AEOBatchRunner(max_concurrent_requests=5)
        
        print("Executing batch test...")
        summary = await batch_runner.run_batch_audit(
            target_brand="Notion",
            target_domain="notion.so",
            category="Note Taking and Workspace Software"
        )
        
        print("\n================ BATCH AUDIT SUMMARY ================")
        print(f"Target Brand: {summary.target_brand}")
        print(f"Prompts Evaluated: {summary.total_prompts_evaluated}")
        print(f"Share of Voice (SoV): {summary.share_of_voice_percentage}%")
        print(f"Average Rank Position: {summary.average_rank_when_mentioned}")
        print("\nTop Competitors Mentioned:")
        for comp, count in summary.competitor_mentions_summary.items():
            print(f" - {comp}: {count} prompts")
        print("\nTop Citation URLs Powering Recommendations:")
        for url in summary.top_citation_urls[:5]:
            print(f" - {url}")
        print("=====================================================")

    asyncio.run(main())