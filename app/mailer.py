"""Password-reset delivery. Uses SMTP when AVASOC_SMTP_HOST is set; otherwise (dev only) logs the link."""
import logging, os, smtplib
from email.message import EmailMessage

log = logging.getLogger("avasoc")


def send_reset(email, token):
    base = os.environ.get("AVASOC_APP_URL", "http://127.0.0.1:8000").rstrip("/"); link = f"{base}/#/reset/{token}"
    host = os.environ.get("AVASOC_SMTP_HOST")
    if not host:
        if os.environ.get("AVASOC_ENV") == "dev": log.warning("DEV password reset link for %s: %s", email, link)
        else: log.error("Password reset requested but AVASOC_SMTP_HOST is not configured; no email was sent.")
        return
    m = EmailMessage(); m["Subject"] = "Reset your AV-ASOC password"; m["From"] = os.environ.get("AVASOC_SMTP_FROM", "no-reply@localhost"); m["To"] = email
    m.set_content(f"Use this link within 30 minutes to reset your password:\n\n{link}\n\nIf you did not ask for this, ignore this email.")
    with smtplib.SMTP(host, int(os.environ.get("AVASOC_SMTP_PORT", "587")), timeout=15) as s:
        s.starttls()
        if os.environ.get("AVASOC_SMTP_USER"): s.login(os.environ["AVASOC_SMTP_USER"], os.environ.get("AVASOC_SMTP_PASS", ""))
        s.send_message(m)
