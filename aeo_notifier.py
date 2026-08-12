import os
import email.utils
from email.message import EmailMessage
import aiosmtplib
from dotenv import load_dotenv

load_dotenv()

class AEONotifier:
    def __init__(self):
        # Sanitize env variables by stripping whitespaces and accidental quotes
        raw_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
        self.smtp_host = raw_host.strip().strip("'").strip('"') if raw_host else "smtp.gmail.com"
        
        raw_port = os.getenv("SMTP_PORT", "587")
        try:
            self.smtp_port = int(raw_port.strip().strip("'").strip('"'))
        except ValueError:
            self.smtp_port = 587

        raw_user = os.getenv("SMTP_USER", "")
        self.smtp_user = raw_user.strip().strip("'").strip('"') if raw_user else ""

        raw_pass = os.getenv("SMTP_PASSWORD", "")
        self.smtp_password = raw_pass.strip().strip("'").strip('"') if raw_pass else ""

        raw_from = os.getenv("ALERT_FROM_EMAIL", self.smtp_user)
        self.from_email = raw_from.strip().strip("'").strip('"') if raw_from else self.smtp_user

    async def send_budgetflow_alert(
        self,
        user_email: str,
        target_brand: str = "BudgetFlow",
        previous_sov: float = 80.0,
        current_sov: float = 63.33,
        lost_prompts: list = None,
        report_url: str = "http://localhost:3000/report/latest"
    ):
        """Sends an urgent email alert when BudgetFlow Share of Voice drops."""
        if lost_prompts is None:
            lost_prompts = [
                "best multi-currency personal finance tracker",
                "free budgeting app with live currency rates",
                "top budgeting tools for young professionals"
            ]

        drop_amount = round(previous_sov - current_sov, 2)
        
        msg = EmailMessage()
        msg["From"] = self.from_email
        msg["To"] = user_email
        msg["Subject"] = f"⚠️ Alert: {target_brand} Share of Voice dropped to {current_sov}% (-{drop_amount}%)"

        # Headers added to optimize inbox deliverability and prevent spam filtering
        msg["Message-ID"] = email.utils.make_msgid(domain="gmail.com")
        msg["Date"] = email.utils.formatdate(localtime=True)
        msg["Auto-Submitted"] = "auto-generated"
        msg["X-Auto-Response-Suppress"] = "All"
        msg["Precedence"] = "bulk"

        lost_items_html = "".join(
            [f"<li style='margin-bottom: 8px;'><b>Missing on Query:</b> <i>\"{p}\"</i></li>" for p in lost_prompts[:5]]
        )

        html_content = f"""
        <html>
            <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background-color: #020617; color: #f8fafc; padding: 24px;">
                <div style="max-width: 600px; margin: 0 auto; background-color: #0f172a; border-radius: 12px; padding: 28px; border: 1px solid #1e293b;">
                    
                    <!-- BRAND HEADER -->
                    <div style="border-bottom: 1px solid #1e293b; padding-bottom: 16px; margin-bottom: 20px;">
                        <span style="font-size: 11px; font-weight: bold; text-transform: uppercase; letter-spacing: 1px; color: #6366f1;">BudgetFlow AEO Intelligence</span>
                        <h2 style="color: #ffffff; margin-top: 4px; margin-bottom: 0; font-size: 20px;">Share of Voice Drop Detected</h2>
                    </div>

                    <p style="color: #94a3b8; font-size: 14px; line-height: 1.5;">
                        During the latest Answer Engine audit scan, <b>{target_brand}</b> experienced a <b>{drop_amount}% drop</b> in organic recommendations across target financial app prompts.
                    </p>
                    
                    <!-- METRIC BOXES -->
                    <div style="background-color: #020617; padding: 16px; border-radius: 8px; margin: 20px 0; border: 1px solid #1e293b;">
                        <table width="100%" cellPadding="0" cellSpacing="0" border="0">
                            <tr>
                                <td align="center" style="padding: 8px;">
                                    <span style="font-size: 11px; color: #64748b; text-transform: uppercase; font-weight: 600;">Previous SoV</span>
                                    <div style="font-size: 22px; font-weight: 800; color: #38bdf8; margin-top: 4px;">{previous_sov}%</div>
                                </td>
                                <td align="center" style="padding: 8px; border-left: 1px solid #1e293b; border-right: 1px solid #1e293b;">
                                    <span style="font-size: 11px; color: #64748b; text-transform: uppercase; font-weight: 600;">Current SoV</span>
                                    <div style="font-size: 22px; font-weight: 800; color: #ef4444; margin-top: 4px;">{current_sov}%</div>
                                </td>
                                <td align="center" style="padding: 8px;">
                                    <span style="font-size: 11px; color: #64748b; text-transform: uppercase; font-weight: 600;">Visibility Loss</span>
                                    <div style="font-size: 22px; font-weight: 800; color: #f59e0b; margin-top: 4px;">-{drop_amount}%</div>
                                </td>
                            </tr>
                        </table>
                    </div>

                    <!-- TOP LOST PROMPTS -->
                    <h4 style="color: #f8fafc; margin-bottom: 12px; font-size: 14px;">High-Value Prompts Currently Missing BudgetFlow:</h4>
                    <ul style="color: #cbd5e1; padding-left: 20px; font-size: 13px; line-height: 1.6;">
                        {lost_items_html}
                    </ul>

                    <!-- ACTION BUTTON -->
                    <div style="margin-top: 32px; text-align: center;">
                        <a href="{report_url}" style="background-color: #4f46e5; color: #ffffff; text-decoration: none; padding: 12px 28px; border-radius: 8px; font-weight: 600; font-size: 14px; display: inline-block;">
                            View Audit Report & Generate Fixes
                        </a>
                    </div>

                </div>
            </body>
        </html>
        """
        msg.add_alternative(html_content, subtype="html")

        if not self.smtp_user or not self.smtp_password:
            print(f"[Simulated Alert] Sent BudgetFlow drop report to {user_email} (From: {self.from_email})")
            return

        try:
            is_port_465 = (self.smtp_port == 465)
            await aiosmtplib.send(
                msg,
                hostname=self.smtp_host,
                port=self.smtp_port,
                username=self.smtp_user,
                password=self.smtp_password,
                use_tls=is_port_465,
                start_tls=(not is_port_465)
            )
            print(f"[Notifier] Email alert successfully sent to {user_email}")
        except Exception as e:
            print(f"[Error] Failed to send email alert: {e}")