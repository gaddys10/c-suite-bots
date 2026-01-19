import os
from typing import List, Tuple

from figma_visual_diff import run_visual_poll
from memory import kv_get, kv_set, write_signal


def _safe_remove(path: str) -> None:
    try:
        os.remove(path)
    except Exception:
        pass


def _is_number(x: object) -> bool:
    return isinstance(x, (int, float))


def upload_diffs_and_signal(
    *,
    slack_client,
    diffs_channel_id: str,
    file_key: str,
    pct_threshold: float = 0.25,
    top_k: int = 8,
) -> List[Tuple[str, float, str]]:
    """
    Runs visual diff poll, uploads meaningful diffs to diffs channel, writes one DB signal.

    Returns: list of (node_id, pct_changed, slack_permalink)
    """
    results = run_visual_poll(file_key)
    if not results:
        return []

    changed: List[Tuple[str, float, str]] = []
    for nid, pct_or_status, diff_path in results:
        if not _is_number(pct_or_status):
            continue
        if not diff_path:
            continue
        pct = float(pct_or_status)
        if pct < pct_threshold:
            _safe_remove(diff_path)
            continue
        changed.append((nid, pct, diff_path))

    if not changed:
        return []

    changed.sort(key=lambda x: x[1], reverse=True)
    changed = changed[:top_k]

    uploaded: List[Tuple[str, float, str]] = []
    for nid, pct, diff_path in changed:
        title = f"Figma diff {nid} (~{pct:.2f}%)"
        resp = slack_client.files_upload_v2(
            channel=diffs_channel_id,
            file=diff_path,
            title=title,
        )
        file_obj = resp.get("file") or {}
        permalink = file_obj.get("permalink") or ""
        uploaded.append((nid, pct, permalink))
        _safe_remove(diff_path)

    # Stable de-dupe fingerprint
    fp = "|".join([f"{nid}:{pct:.2f}" for nid, pct, _ in uploaded])
    last_fp = kv_get("figma:last_upload_fp", "")
    if fp and fp != last_fp:
        kv_set("figma:last_upload_fp", fp)

        lines = []
        for nid, pct, link in uploaded:
            if link:
                lines.append(f"- node {nid}: ~{pct:.2f}% (diff: {link})")
            else:
                lines.append(f"- node {nid}: ~{pct:.2f}%")

        write_signal(
            channel="system",
            role="Council",
            kind="figma_event",
            text="Figma visual diffs uploaded.\n" + "\n".join(lines),
        )

    return uploaded
