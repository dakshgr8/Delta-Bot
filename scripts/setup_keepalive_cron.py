#!/usr/bin/env python3
"""
scripts/setup_keepalive_cron.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Configures and verifies the 24/7 Render keep-alive cron job via cron-job.org API v2.
Prevents Render free web services from sleeping after 15 minutes of inactivity.
"""

import os
import sys
import httpx
from pathlib import Path

def main() -> None:
    # Load .env if present
    env_file = Path(__file__).resolve().parent.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

    api_key = os.environ.get("CRON_JOB_ORG_API_KEY", "").strip()
    if not api_key:
        print("ERROR: CRON_JOB_ORG_API_KEY not set in environment or .env")
        sys.exit(1)

    backend_url = os.environ.get("RENDER_BACKEND_URL", "https://delta-bot-07e2.onrender.com").rstrip("/")
    target_url = f"{backend_url}/api/status"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    print(f"Target Keep-Alive URL: {target_url}")

    # Check existing jobs
    res = httpx.get("https://api.cron-job.org/jobs", headers=headers)
    if res.status_code != 200:
        print(f"ERROR: Failed to list jobs: {res.status_code} {res.text}")
        sys.exit(1)

    jobs = res.json().get("jobs", [])
    delta_job = None
    for j in jobs:
        if "delta-bot" in j.get("url", "").lower() or "delta-bot" in j.get("title", "").lower():
            delta_job = j
            break

    if delta_job:
        print(f"Existing Delta-Bot Keep-Alive Job found (ID: {delta_job['jobId']})")
        print(f"  Title: {delta_job['title']}")
        print(f"  URL: {delta_job['url']}")
        print(f"  Enabled: {delta_job['enabled']}")
        print(f"  Schedule: Every 10 minutes")
    else:
        print("Creating new Keep-Alive Job on cron-job.org...")
        payload = {
            "job": {
                "url": target_url,
                "title": "Delta-Bot Render Keep-Alive",
                "enabled": True,
                "saveResponses": True,
                "schedule": {
                    "timezone": "UTC",
                    "hours": [-1],
                    "mdays": [-1],
                    "minutes": [0, 10, 20, 30, 40, 50],
                    "months": [-1],
                    "wdays": [-1],
                },
            }
        }
        create_res = httpx.put("https://api.cron-job.org/jobs", headers=headers, json=payload)
        if create_res.status_code in (200, 201):
            data = create_res.json()
            print(f"SUCCESS: Created Keep-Alive job with ID {data.get('jobId')}")
        else:
            print(f"ERROR: Failed to create job: {create_res.status_code} {create_res.text}")

if __name__ == "__main__":
    main()
