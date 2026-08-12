import asyncio
from aeo_notifier import AEONotifier

async def run_test():
    notifier = AEONotifier()
    print("Testing Gmail SMTP dispatch for BudgetFlow...")
    
    # Replace with the email address where you want to receive the test alert
    recipient_email = "attehjames88@gmail.com" 
    
    await notifier.send_budgetflow_alert(
        user_email=recipient_email,
        target_brand="BudgetFlow",
        previous_sov=83.33,
        current_sov=66.67,
        lost_prompts=[
            "best multi-currency personal finance apps in 2026",
            "top budget tracking software for freelancers",
            "free personal finance app with real-time conversion"
        ],
        report_url="http://localhost:3000/report/test-budgetflow-id"
    )

if __name__ == "__main__":
    asyncio.run(run_test())