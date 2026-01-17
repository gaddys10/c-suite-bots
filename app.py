from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from openai import OpenAI
import os
from dotenv import load_dotenv
from datetime import datetime, timezone
import time
from typing import Dict, Tuple, List
from collections import deque

# ---- Cross-channel awareness buffer (in-memory) ----
# NOTE: resets on restart/deploy. We'll persist later (Redis/Postgres).
EVENT_BUFFER_MAX = 500
EVENT_BUFFER = deque(maxlen=EVENT_BUFFER_MAX)

# Per-role-channel "last brief" timestamp so briefs are incremental
LAST_BRIEF_TS_BY_CHANNEL: Dict[str, float] = {}

load_dotenv()

# -- OpenAI Client --
key = os.getenv("OPENAI_API_KEY", "")
if not key:
    raise RuntimeError("OPENAI_API_KEY missing")
oai = OpenAI(api_key=key)
# oai = OpenAI()

# -- Slack App --
app = App(token=os.environ["SLACK_BOT_TOKEN"])

#
BOT_USER_ID = app.client.auth_test()["user_id"]

# -- Role mapping to Slack channels ---
ROLE_MAP = {
    "chief-of-staff": "Chief of Staff",
    "chief-product-officer": "Chief Product Officer",
    "chief-technology-officer": "Chief Technology Officer",
    "chief-strategy-officer": "Chief Strategy Officer",
    "chief-risk-officer": "Chief Risk Officer",
    "council": "Council",
    "bot-sandbox": "Chief Risk Officer",  # for testing purposes
}

#--- System prompt template for each role ---
SYSTEM_PROMPT = """
You are operating as an executive role inside Gaddico.

Your role is determined strictly by the Slack channel you are responding in.

You MUST:
- Follow the pinned manifest in this channel exactly
- Stay within the Allowed / Not Allowed boundaries
- Refuse any request that violates the manifest
- Explain refusals briefly and clearly

You MUST NOT:
- Offer advice outside your role
- Blend roles
- Brainstorm broadly
- Ignore constraints

If the channel is #council:
- Respond role by role when explicitly asked
- Never merge perspectives unless requested

Your job is decision clarity, not agreement.
""".strip()

PROJECT_KEYWORDS = {
    "BoxWeb": ["boxweb", "fighter", "rank", "compare", "p4p", "division"],
    "Bookity": ["bookity", "booking", "appointment", "service", "business"],
    "Vovis": ["vovis", "resume", "job", "career", "ats"],
    "Consonant": ["consonant", "consonant software"],
    "Gaddico": ["gaddico", "holding", "c-suite", "c suite", "council"],
}

AWARENESS_RULE = (
    "Cross-channel awareness is permitted for internal reasoning only. "
    "Do not quote or reveal message text from other channels, do not identify channel names, "
    "and do not identify authors. Only discuss abstract implications/signals and next actions."
)

def build_signal_summary(since_ts: float) -> str:
    items = [x for x in list(EVENT_BUFFER) if x["ts"] > since_ts]
    if not items:
        return "No new workspace signals since the last brief."

    # compress into a small textual blob (still no quotes, no channels, no authors)
    lines = []
    for x in items[-80:]:  # cap
        s = x["signal"]
        topics = ",".join(s["topics"]) if s["topics"] else "Unknown"
        flags = []
        if s["is_question"]: flags.append("question")
        if s["mentions_blocker"]: flags.append("blocker")
        if s["mentions_decision"]: flags.append("decision")
        if s["mentions_ship"]: flags.append("shipping")
        lines.append(f"- topics={topics}; flags={','.join(flags) if flags else 'none'}; size={s['len']}")
    return "\n".join(lines)


def extract_signals(text: str) -> Dict[str, object]:
    t = (text or "").lower()
    topics = []
    for topic, kws in PROJECT_KEYWORDS.items():
        if any(k in t for k in kws):
            topics.append(topic)

    signal = {
        "topics": topics[:3],
        "is_question": "?" in t,
        "mentions_blocker": any(k in t for k in ["blocked", "stuck", "can't", "error", "failing", "issue"]),
        "mentions_decision": any(k in t for k in ["decide", "decision", "choose", "should we", "approve"]),
        "mentions_ship": any(k in t for k in ["ship", "deploy", "release", "launch"]),
        "len": len(t),
    }
    return signal

def record_event(body: dict):
    event = body.get("event") or {}

    # ignore bot messages + edits/etc
    if event.get("bot_id") or event.get("subtype"):
        return

    text = (event.get("text") or "").strip()
    if not text:
        return

    ts = float(event.get("ts") or time.time())
    signal = extract_signals(text)

    # DO NOT store channel name or user id for Option A.
    EVENT_BUFFER.append({
        "ts": ts,
        "signal": signal,
    })


# ----------------------------
# Pinned-manifest cache helpers
# ----------------------------

# Simple in-memory cache with TTL
# TTL means we re-fetch pinned messages every N seconds
_manifest_cache: Dict[str, Tuple[float, str]] = {}
MANIFEST_TTL_SECONDS = 300  # 5 minutes

def fetch_recent_messages(channel_id: str, limit: int = 20) -> str:
    """
    Minimal history fetch: grabs the most recent N messages from a channel.
    """
    resp = app.client.conversations_history(channel=channel_id, limit=limit)
    msgs = resp.get("messages", []) or []

    lines = []
    for m in reversed(msgs):  # oldest -> newest
        if m.get("bot_id") or m.get("subtype") == "bot_message":
            continue
        text = (m.get("text") or "").strip()
        if not text:
            continue
        user = m.get("user", "unknown")
        lines.append(f"{user}: {text}")

    return "\n".join(lines) if lines else "No recent human messages found."

def _fetch_pinned_messages_text(channel_id: str) -> List[str]:
    """
    Returns a list of text blocks from pinned *message* items in this channel.
    """
    pins = app.client.pins_list(channel=channel_id)
    items = pins.get("items", [])

    texts: List[str] = []
    for item in items:
        if item.get("type") != "message":
            continue
        msg = item.get("message", {})
        text = (msg.get("text") or "").strip()
        if text:
            texts.append(text)

    return texts

def get_channel_manifest(channel_id: str) -> str:
    """
    Cached fetch of pinned manifest text for a channel.
    Joins multiple pinned messages with separators (your screenshots show this is common).
    """
    now = time.time()
    cached = _manifest_cache.get(channel_id)
    if cached and cached[0] > now:
        return cached[1]

    try:
        pinned_texts = _fetch_pinned_messages_text(channel_id)
        if not pinned_texts:
            manifest = "No pinned manifest found for this channel."
        else:
            manifest = "\n\n---\n\n".join(pinned_texts)

        # Safety cap to prevent prompt blowups if someone pins a book.
        MAX_CHARS = 12000
        if len(manifest) > MAX_CHARS:
            manifest = manifest[:MAX_CHARS] + "\n\n[TRUNCATED: manifest too long]"
    except Exception as e:
        manifest = f"Manifest could not be retrieved due to an error: {e!r}"

    _manifest_cache[channel_id] = (now + MANIFEST_TTL_SECONDS, manifest)
    return manifest

# --- Helper function to retrieve channel names ---
def get_channel_name(channel_id: str) -> str:
    """Best-effort channel name lookup."""
    try:
        # Use conversations.info to fetch channel information
        info = app.client.conversations_info(channel=channel_id)
        return info["channel"]["name"]
    except Exception:
        return "unknown"

def build_system_prompt(role: str, channel_manifest: str) -> str:
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"You are acting as: {role}.\n\n"
        f"CHANNEL MANIFEST (PINNED):\n{channel_manifest}\n\n"
        f"ENFORCEMENT RULES:\n"
        f"- If the request conflicts with the manifest, refuse and cite the specific section.\n"
        f"- Do not invent constraints not present in the manifest.\n"
        f"- Keep answers concise, direct, and role-consistent.\n"
    )


def run_llm(role: str, channel_manifest: str, user_text: str) -> str:
    sys = build_system_prompt(role, channel_manifest)

    resp = oai.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": user_text},
        ],
    )
    return (resp.choices[0].message.content or "").strip()

#---- Slack Event Handlers ---
# Handle app mentions
@app.event("app_mention")
def handle_app_mention(body, event, say, logger):
    # record signals too (mentions count)
    record_event(body)

    channel_id = event["channel"]
    text = (event.get("text") or "").strip()
    if not text:
        return

    channel_name = get_channel_name(channel_id)
    role = ROLE_MAP.get(channel_name, "Unknown")
    channel_manifest = get_channel_manifest(channel_id)

    answer = run_llm(role, channel_manifest, text)
    if answer:
        say(answer)


# Optional: if you want the bot to reply to any message (not just mentions),
# uncomment this and decide how you want to scope it.
@app.event("message")
def handle_message_events(body, event, say, logger):
    record_event(body)

    channel_id = event.get("channel")
    text = (event.get("text") or "").strip()
    if not channel_id or not text:
        return

    # Ignore bot messages + message subtypes (edits, joins, etc.)
    if event.get("bot_id") or event.get("subtype"):
        return

    channel_name = get_channel_name(channel_id)
    is_role_channel = channel_name in ROLE_MAP
    channel_type = event.get("channel_type")  # channel/group/im/mpim
    is_dm = channel_type == "im"

    # Outside role channels: only respond if DM or explicitly mentioned
    if not is_role_channel and not is_dm:
        if f"<@{BOT_USER_ID}>" not in text:
            return

    # Cross-channel awareness brief (Option A compliant)
    if text.lower() in ["brief me", "what changed today", "daily brief"]:
        since = LAST_BRIEF_TS_BY_CHANNEL.get(channel_id, 0.0)
        signals = build_signal_summary(since_ts=since)
        LAST_BRIEF_TS_BY_CHANNEL[channel_id] = time.time()

        role = ROLE_MAP.get(channel_name, "Unknown")
        channel_manifest = get_channel_manifest(channel_id)

        prompt = (
            f"{AWARENESS_RULE}\n\n"
            f"WORKSPACE SIGNALS (ABSTRACTED):\n{signals}\n\n"
            f"Return:\n"
            f"1) 3-7 key signals\n"
            f"2) 1-3 recommended next actions for this role\n"
            f"3) 0-2 risks/watchouts\n"
        )
        answer = run_llm(role, channel_manifest, prompt)
        if answer:
            say(answer)
        return

    # Normal role response
    role = ROLE_MAP.get(channel_name, "Unknown")
    channel_manifest = get_channel_manifest(channel_id)
    answer = run_llm(role, channel_manifest, text)
    if answer:
        say(answer)

# --- Start the app ---

if __name__ == "__main__":
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()
