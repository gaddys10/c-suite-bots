# Gaddico Council Bot

Slack bot implementing role-constrained executive perspectives:
- Chief of Staff
- CPO
- CTO
- CSO
- CRO

Roles are enforced by channel context and pinned manifests.

Run locally using Socket Mode.

python -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt

.env keys:

SLACK_BOT_TOKEN=...

SLACK_APP_TOKEN=...

OPENAI_API_KEY=...

python app.py