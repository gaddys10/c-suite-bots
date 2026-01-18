from pydoc import text
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from openai import OpenAI
import os
from dotenv import load_dotenv
from datetime import datetime, timezone
import time
import sys
from typing import Dict, Tuple, List
from collections import deque
from apscheduler.schedulers.background import BackgroundScheduler
from memory import init_db, kv_get, kv_set, write_signal, get_signals_since, get_last_brief_ts, set_last_brief_ts
from github_read import recent_open_prs, commit_summary, compare_commits, recent_commit_shas

init_db()

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

# -- Slack App --
app = App(token=os.environ["SLACK_BOT_TOKEN"])

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

def build_channel_id_map():
    resp = app.client.conversations_list(limit=1000)
    chans = resp.get("channels", []) or []
    return {c["name"]: c["id"] for c in chans}


CHANNEL_ID_BY_NAME = build_channel_id_map()
GENERAL_CHANNEL_ID = CHANNEL_ID_BY_NAME.get("general")

# --- GitHub repos to track ---
TRACKED_REPOS = [
    "gaddys10/boxweb-site",
    "gaddys10/appointment-booker",
    "gaddys10/jobari-engine",
    "gaddys10/boxing-gym-finder",
    "gaddys10/c-suite-bots",
    "gaddys10/home-site",
]

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
    "BoxWeb": ["boxweb", "boxing", "fighter", "rank", "compare", "p4p", "division"],
    "Bookity": ["bookity", "booking", "appointment", "service", "business"],
    "Vovis": ["vovis", "resume", "job", "career", "ats",],
    "Consonant": ["consonant", "consonant software"],
    "Gaddico": ["gaddico", "gaddy", "locklear", "gaddy locklear", "Ventures", "holding", "holdings", "c-suite", "c suite", "council", "councils"],
}

AWARENESS_RULE = (
    "Cross-channel awareness is permitted for internal reasoning only. "
    "Do not quote or reveal message text from other channels, do not identify channel names, "
    "and do not identify authors. Only discuss abstract implications/signals and next actions."
)

def set_focus_from_text(text: str):
    raw = (text or "").strip()
    if not raw.lower().startswith("focus:"):
        return False

    focus_text = raw.split(":", 1)[1].strip()
    if not focus_text:
        return False

    sig = extract_signals(focus_text)
    topics = sig.get("topics") or []

    kv_set("focus:text", focus_text)
    kv_set("focus:topics", ",".join(topics))
    kv_set("focus:set_at", str(time.time()))
    return True

def refresh_auto_focus():
    now = time.time()
    focus_text, topics = infer_focus_from_recent_activity(now)

    if focus_text:
        kv_set("focus:auto:text", focus_text)
        kv_set("focus:auto:topics", ",".join(topics))
        kv_set("focus:auto:set_at", str(now))

def infer_focus_from_recent_activity(now_ts: float) -> tuple[str, list[str]]:
    """
    Infer focus from the last ~90 minutes of signals + workspace event buffer.
    Returns (focus_text, focus_topics).
    """
    WINDOW = 90 * 60

    # Use DB signals (persistent) as primary source
    rows = get_signals_since(now_ts - WINDOW)

    # Score topics
    topic_score = {}
    recent_todos = []
    recent_blockers = 0

    for _, _, role, kind, text in rows:
        sig = extract_signals(text or "")
        for t in sig.get("topics") or []:
            topic_score[t] = topic_score.get(t, 0) + 1

        if sig.get("mentions_ship"):
            for t in sig.get("topics") or []:
                topic_score[t] = topic_score.get(t, 0) + 2

        if sig.get("mentions_decision"):
            for t in sig.get("topics") or []:
                topic_score[t] = topic_score.get(t, 0) + 2

        if sig.get("mentions_blocker"):
            recent_blockers += 1

        if (text or "").strip().startswith("TODO:"):
            recent_todos.append((text or "").strip()[5:].strip())

    # Pick top topics
    ranked = sorted(topic_score.items(), key=lambda x: x[1], reverse=True)
    top_topics = [t for t, _ in ranked[:2]]

    if not ranked and not recent_todos:
        return ("", [])

    # Create a human focus sentence
    if recent_todos:
        focus_text = f"Today’s focus appears to be: {recent_todos[-1][:140]}"
    else:
        focus_text = f"Today’s focus appears to be: {', '.join(top_topics)}"

    if recent_blockers >= 2:
        focus_text += " (watch: blockers showing up repeatedly)"

    return (focus_text, top_topics)

def drift_check():
    # --- config ---
    WINDOW_SECONDS = 30 * 60        # look at last 30 min
    COOLDOWN_SECONDS = 45 * 60      # don’t nag more often than this
    MIN_EVENTS = 6                  # need enough signals to judge

    now = time.time()

    # cooldown
    last_nudge = float(kv_get("drift:last_nudge_ts", "0") or 0)
    if now - last_nudge < COOLDOWN_SECONDS:
        return

    focus_text = kv_get("focus:auto:text", "").strip()
    focus_topics = [t for t in (kv_get("focus:auto:topics", "") or "").split(",") if t]

    if not focus_text:
        return  # no inferred focus yet

    # collect recent abstract signals (no channels, no authors, no quotes)
    since = now - WINDOW_SECONDS
    recent = [x for x in list(EVENT_BUFFER) if x["ts"] >= since]
    if len(recent) < MIN_EVENTS:
        return

    # count topic overlap
    total = 0
    on_focus = 0
    blockers = 0
    decisions = 0

    for x in recent:
        s = x["signal"]
        total += 1
        topics = s.get("topics") or []
        if any(t in focus_topics for t in topics):
            on_focus += 1
        if s.get("mentions_blocker"):
            blockers += 1
        if s.get("mentions_decision"):
            decisions += 1

    ratio = on_focus / max(total, 1)

    # drift rules (simple + effective)
    drifted = (ratio < 0.35)  # <35% of signals match focus topics
    urgent = (blockers >= 2) or (decisions >= 2)

    if not drifted and not urgent:
        return

    # craft a role-style nudge from Chief of Staff (no quotes / no channel names)
    cos_channel_id = CHANNEL_ID_BY_NAME.get("chief-of-staff")
    if not cos_channel_id:
        return

    manifest = get_channel_manifest(cos_channel_id)

    signals_blob = build_signal_summary(since_ts=since)

    prompt = (
        f"{AWARENESS_RULE}\n\n"
        f"Founder focus:\n- {focus_text}\n\n"
        f"Last 30 minutes of WORKSPACE SIGNALS (ABSTRACTED):\n{signals_blob}\n\n"
        f"Your job: interrupt drift.\n"
        f"Write ONE short message to Syrus:\n"
        f"- If drifted: call it out and give 1 concrete redirect step.\n"
        f"- If blockers/decisions: ask 1 sharp question to unblock.\n"
        f"- Be direct, human, not robotic. No acknowledgements.\n"
    )

    msg = run_llm("Chief of Staff", manifest, prompt)
    if msg and msg.strip():
        app.client.chat_postMessage(channel=cos_channel_id, text=msg.strip())
        kv_set("drift:last_nudge_ts", str(now))

def format_commit_event(repo: str, sha: str) -> str:
    s = commit_summary(repo, sha)
    files = s.get("files", []) or []
    file_lines = ", ".join(
        f"{f['filename']} (+{f['additions']}/-{f['deletions']})" for f in files[:8]
    )
    if len(files) > 8:
        file_lines += f", +{len(files)-8} more"

    stats = s.get("stats") or {}
    return (
        f"{repo} commit {sha[:7]} — {s.get('message','')}\n"
        f"Author: {s.get('author')} @ {s.get('date')}\n"
        f"Stats: +{stats.get('additions',0)}/-{stats.get('deletions',0)} across {len(files)} files\n"
        f"Files: {file_lines}"
    )

def nudge_roles_about_github(event_text: str):
    parts = []
    for ch_name, r in ROLE_MAP.items():
        if r == "Council":
            continue
        channel_id = CHANNEL_ID_BY_NAME.get(ch_name)
        if not channel_id:
            continue
        manifest = get_channel_manifest(channel_id)

        prompt = (
            f"New GitHub event:\n{event_text}\n\n"
            f"As {r}, respond with: (1) implication, (2) next action for Syrus, (3) risk.\n"
            f"Be short."
        )
        resp = run_llm(r, manifest, prompt)
        if resp:
            parts.append(f"*{r}*\n{resp}")

    app.client.chat_postMessage(
        channel="council",
        text="*GitHub Role Nudges*\n\n" + "\n\n".join(parts)
    )

def poll_github():
    
    for repo in TRACKED_REPOS:
        try:
            last_sha = kv_get(f"gh:{repo}:sha", "")
            shas = recent_commit_shas(repo, limit=10)  # newest -> oldest

            if not last_sha and shas:
                kv_set(f"gh:{repo}:sha", shas[0])
                shas = []

            to_process = []
            for s in shas:
                if s == last_sha:
                    break
                to_process.append(s)

            # 🔁 THIS LOOP REPLACES `if sha and sha != last_sha`
            for sha in reversed(to_process):  # oldest -> newest
                event_text = format_commit_event(repo, sha)

                write_signal(
                    channel="system",
                    role="Council",
                    kind="github_event",
                    text=event_text
                )

                for ch_name, role in ROLE_MAP.items():
                    decision_key = f"ruled:commit:{repo}:{sha}:{role}"
                    if kv_get(decision_key):
                        continue
                    if role == "Council":
                        continue

                    channel_id = CHANNEL_ID_BY_NAME.get(ch_name)
                    if not channel_id:
                        continue

                    manifest = get_channel_manifest(channel_id)

                    resp = run_llm(
                        role,
                        manifest,
                        f"A code change just happened:\n{event_text}\n\n"
                        "If you believe the founder (Syrus) needs to know anything about this change or items/events/circumstances associated with it, say so briefly. "
                        "if you believe that this code has made a significant impact to your role/responsibilities or progress concerning them, say so briefly. "
                        "If neither applies, output an EMPTY STRING. Do not acknowledge. Do not say ‘no action needed.’"
                    )

                    if resp and resp.strip():
                        app.client.chat_postMessage(channel=channel_id, text=resp.strip())

                    kv_set(decision_key, "1")

            # ✅ update last seen commit to newest
            if shas:
                kv_set(f"gh:{repo}:sha", shas[0])

            last_pr_fp = kv_get(f"gh:{repo}:pr", "")

            prs = recent_open_prs(repo, limit=10)  # newest -> oldest
            fps = [f"{p['number']}|{p['updated_at']}" for p in prs if p.get("updated_at")]

            # first-run PR guard (mirrors your commit guard behavior)
            if not last_pr_fp and fps:
                kv_set(f"gh:{repo}:pr", fps[0])
                fps = []
            to_process = []
            for fp in fps:
                if fp == last_pr_fp:
                    break
                to_process.append(fp)

            # oldest -> newest
            for fp in reversed(to_process):
                pr = next((p for p in prs if f"{p['number']}|{p['updated_at']}" == fp), None)
                if not pr:
                    continue

                event_text = (
                    f"{repo} PR #{pr['number']} updated — {pr['title']}\n"
                    f"Updated: {pr['updated_at']}\n"
                    f"URL: {pr['html_url']}\n"
                    f"Body: {(pr['body'] or '')[:600]}"
                )

                write_signal(channel="system", role="Council", kind="github_event", text=event_text)

                for ch_name, role in ROLE_MAP.items():
                    if role == "Council":
                        continue

                    decision_key = f"ruled:pr:{repo}:{fp}:{role}"
                    if kv_get(decision_key):
                        continue

                    channel_id = CHANNEL_ID_BY_NAME.get(ch_name)
                    if not channel_id:
                        continue

                    manifest = get_channel_manifest(channel_id)

                    resp = run_llm(
                        role,
                        manifest,
                        f"A code change just happened:\n{event_text}\n\n"
                        "If you believe Syrus needs to know anything related to the code change or items/events/circumstances associated with it, say so briefly. "
                        "If not, output an EMPTY STRING. Do not acknowledge. Do not say ‘no action needed.’"
                    )

                    if resp and resp.strip():
                        app.client.chat_postMessage(channel=channel_id, text=resp.strip())

                    kv_set(decision_key, "1")

            # update last seen PR fp to newest
            if fps:
                kv_set(f"gh:{repo}:pr", fps[0])


        except Exception as e:
            write_signal(
                channel="system",
                role="Council",
                kind="github_error",
                text=f"{repo}: {repr(e)}"
            )
            continue

def run_daily_brief():
    last_ts = get_last_brief_ts()
    rows = get_signals_since(last_ts)

    if not rows:
        summary_input = "No new activity since the last brief."
    else:
        summary_input = "\n".join(
            f"- [{kind.upper()}] ({role}) {text}"
            for _, _, role, kind, text in rows
        )

    briefs = []

    for channel_name, role in ROLE_MAP.items():
        if role == "Council":
            continue  # skip council, it's an aggregator

        role_channel_id = CHANNEL_ID_BY_NAME.get(channel_name)
        if not role_channel_id:
            continue
        manifest = get_channel_manifest(role_channel_id)

        prompt = f"""
You are preparing a PRIORITY BRIEF for the founder.

Based ONLY on the activity below, produce:
1) ONE top priority Syrus must address next
2) Why this priority matters now (1 sentence)
3) ONE risk or blocker that could slow it down
4) ONE concrete next action Syrus should take today

Rules:
- Be decisive, not exhaustive
- Do NOT summarize activity
- Do NOT list multiple options
- If nothing rises to this level, output EMPTY STRING

ACTIVITY:
{summary_input}
""".strip()

        response = run_llm(role, manifest, prompt)
        if response:
            briefs.append(f"*{role}*\n{response}")

    final_brief = "\n\n".join(briefs)

    app.client.chat_postMessage(
        channel="council-briefs",
        text=f"*Executive Priority Brief*\n\n{final_brief}"
    )

    set_last_brief_ts(time.time())

def run_scheduled_jobs():
    scheduler = BackgroundScheduler(timezone="UTC")

    # poll GitHub every 5 minutes
    scheduler.add_job(
        poll_github,
        trigger="interval",
        minutes=5,
        id="poll_github",
        replace_existing=True,
    )

    scheduler.add_job(
        refresh_auto_focus,
        trigger="interval",
        minutes=10,
        id="refresh_auto_focus",
        replace_existing=True,
    )

    scheduler.add_job(
        drift_check,
        trigger="interval",
        minutes=15,
        id="drift_check",
        replace_existing=True,
    )

    # daily brief (optional — delete this block if you don’t want it yet)
    scheduler.add_job(
        run_daily_brief,
        trigger="cron",
        day_of_week="mon,wed,fri",
        hour=13,  # 9am ET = 13 UTC
        minute=0,
        id="daily_brief",
        replace_existing=True,
    )

    scheduler.start()

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

# Simple in-memory cache with TTL (refetching pins every N seconds).
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

        # Safety cap to prevent prompt blowups if someone pins a book
        MAX_CHARS = 12000
        if len(manifest) > MAX_CHARS:
            manifest = manifest[:MAX_CHARS] + "\n\n[TRUNCATED: manifest too long]"
    except Exception as e:
        manifest = f"Manifest could not be retrieved due to an error: {e!r}"

    _manifest_cache[channel_id] = (now + MANIFEST_TTL_SECONDS, manifest)
    return manifest

# --- Helper function to retrieve channel names ---
def get_channel_name(channel_id: str) -> str:
    try:
        # Use conversations_info to fetch channel information
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


# So the bot replies to any message (not just mentions),
@app.event("message")
def handle_message_events(body, event, say, logger):
    event = body.get("event", {}    )
    channel_id = event.get("channel")
    text = (event.get("text") or "").strip()
    if not channel_id or not text:
        return
    
    if set_focus_from_text(text):
        say("Locked. I’ll nudge you if your activity drifts off this focus.")
        return

    # Ignore bot messages + message subtypes (edits, joins, etc.)
    if event.get("bot_id") or event.get("subtype"):
        return

    record_event(body)
    
    # CASE 1: message posted in #general (bulletin board)
    if GENERAL_CHANNEL_ID and channel_id == GENERAL_CHANNEL_ID:
        logger.info("[general] bulletin received")

        bulletin = text

        # Optional: store bulletin as memory signal (keeps it in your DB)
        write_signal(
            channel="general",
            role="Founder",
            kind="bulletin",
            text=bulletin
        )

        # Each role reads bulletin and responds (ONLY) in their own role channel
        for ch_name, role in ROLE_MAP.items():
            if role == "Council":
                continue

            role_channel_id = CHANNEL_ID_BY_NAME.get(ch_name)
            if not role_channel_id:
                continue

            manifest = get_channel_manifest(role_channel_id)

            resp = run_llm(
                role,
                manifest,
                "A new bulletin was posted in #general:\n\n"
                f"{bulletin}\n\n"
                "If you believe Syrus should hear from you about this, reply briefly. "
                "If not, output an EMPTY STRING. Do not acknowledge. Do not say ‘no action needed.’"
            )

            if resp and resp.strip():
                app.client.chat_postMessage(
                    channel=role_channel_id,
                    text=resp.strip()
                )

        return

    channel_name = get_channel_name(channel_id)
    is_role_channel = channel_name in ROLE_MAP
    channel_type = event.get("channel_type")  # channel/group/im/mpim
    is_dm = channel_type == "im"

    # Outside role channels: only respond if DM or explicitly mentioned
    if not is_role_channel and not is_dm:
        if f"<@{BOT_USER_ID}>" not in text:
            return

    # Cross-channel awareness brief
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

    text = event.get("text", "")

    kind = "message"
    if text.startswith("DECISION:"):
        kind = "decision"
    elif text.startswith("RISK:"):
        kind = "risk"
    elif text.startswith("TODO:"):
        kind = "todo"

    write_signal(
        channel=channel_name,
        role=role,
        kind=kind,
        text=event.get("text", "")
    )

    answer = run_llm(role, channel_manifest, text)
    if answer:
        say(answer)

# --- Start the app ---
if __name__ == "__main__":
    run_scheduled_jobs()
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()