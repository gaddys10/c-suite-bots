import os, json, time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import requests
from PIL import Image, ImageChops

FIGMA_BASE = "https://api.figma.com"
STATE_PATH = os.getenv("FIGMA_STATE_PATH", "figma_state.json")
WATCH_PATH = os.getenv("FIGMA_WATCH_PATH", "figma_watch.json")

@dataclass
class FigmaVersion:
    id: str
    created_at: str

class FigmaClient:
    def __init__(self, token: Optional[str] = None) -> None:
        self.token = (token or os.environ["FIGMA_TOKEN"]).strip()
        self.s = requests.Session()
        self.s.headers.update({"X-Figma-Token": self.token, "Accept": "application/json"})

    def get_versions(self, file_key: str) -> Dict:
        r = self.s.get(f"{FIGMA_BASE}/v1/files/{file_key}/versions", timeout=30)
        r.raise_for_status()
        return r.json()

    def get_images(
        self,
        file_key: str,
        node_ids: List[str],
        *,
        version: Optional[str] = None,
        scale: float = 2.0,
        fmt: str = "png",
        contents_only: bool = True,
        use_absolute_bounds: bool = False,
    ) -> Dict[str, Optional[str]]:
        params = {
            "ids": ",".join(node_ids),
            "format": fmt,
            "scale": scale,
            "contents_only": str(contents_only).lower(),
            "use_absolute_bounds": str(use_absolute_bounds).lower(),
        }
        if version:
            params["version"] = version  # supported by /v1/images/:key :contentReference[oaicite:6]{index=6}

        r = self.s.get(f"{FIGMA_BASE}/v1/images/{file_key}", params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        return data.get("images", {})  # map node_id -> url (may be null) :contentReference[oaicite:7]{index=7}

def load_json(path: str, default):
    if not os.path.exists(path):
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def download_png(url: str, out_path: str) -> None:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        f.write(r.content)

def diff_images(a_path: str, b_path: str, out_path: str) -> Tuple[float, Tuple[int,int,int,int] | None]:
    a = Image.open(a_path).convert("RGBA")
    b = Image.open(b_path).convert("RGBA")

    # normalize size (Figma renders can differ slightly)
    w = min(a.size[0], b.size[0])
    h = min(a.size[1], b.size[1])
    a = a.crop((0, 0, w, h))
    b = b.crop((0, 0, w, h))

    d = ImageChops.difference(a, b)
    bbox = d.getbbox()  # None if identical
    if bbox is None:
        # 0% changed, still save a tiny diff for “no change”
        d.save(out_path)
        return 0.0, None

    # percent changed (rough): count non-zero alpha pixels in diff
    # (fast enough for UI screenshots)
    diff_pixels = 0
    total = w * h
    px = d.getdata()
    for (r, g, bl, al) in px:
        if r or g or bl:
            diff_pixels += 1
    pct = (diff_pixels / total) * 100.0
    d.save(out_path)
    return pct, bbox

def get_latest_version_id(versions_json: Dict) -> Optional[str]:
    versions = versions_json.get("versions", [])
    if not versions:
        return None
    # Figma returns newest-first in practice; still be safe:
    # pick first item’s id
    return versions[0].get("id")

def run_visual_poll(file_key: str, out_dir: str = "/tmp/figma_renders"):
    os.makedirs(out_dir, exist_ok=True)
    state = load_json(STATE_PATH, {"last_version_id": None})
    watch = load_json(WATCH_PATH, {"node_ids": []})

    # dedupe node ids while keeping order
    raw_ids: List[str] = watch.get("node_ids", [])
    seen = set()
    node_ids: List[str] = []
    for nid in raw_ids:
        if nid in seen:
            continue
        seen.add(nid)
        node_ids.append(nid)

    if not node_ids:
        raise RuntimeError(f"No node_ids in {WATCH_PATH}. Add frames to watch first.")

    fc = FigmaClient()
    latest_versions = fc.get_versions(file_key)
    new_version = get_latest_version_id(latest_versions)
    old_version = state.get("last_version_id")

    if not new_version:
        return []

    if old_version == new_version:
        return []

    old_urls = fc.get_images(file_key, node_ids, version=old_version) if old_version else {}
    new_urls = fc.get_images(file_key, node_ids, version=new_version)

    results = []
    ts = int(time.time())

    for nid in node_ids:
        new_url = new_urls.get(nid)
        if not new_url:
            results.append((nid, "render_failed_new", None))
            continue

        new_path = os.path.join(out_dir, f"{nid.replace(':','_')}_new_{ts}.png")
        download_png(new_url, new_path)

        if not old_version:
            # first snapshot: no diff yet
            results.append((nid, "first_snapshot", None))
            try: os.remove(new_path)
            except: pass
            continue

        old_url = old_urls.get(nid)
        if not old_url:
            results.append((nid, "render_failed_old", None))
            try: os.remove(new_path)
            except: pass
            continue

        old_path = os.path.join(out_dir, f"{nid.replace(':','_')}_old_{ts}.png")
        download_png(old_url, old_path)

        diff_path = os.path.join(out_dir, f"{nid.replace(':','_')}_diff_{ts}.png")
        pct, _bbox = diff_images(old_path, new_path, diff_path)
        results.append((nid, pct, diff_path))

        # cleanup old/new, keep diff for upload
        try: os.remove(old_path)
        except: pass
        try: os.remove(new_path)
        except: pass

    state["last_version_id"] = new_version
    save_json(STATE_PATH, state)

    return results

if __name__ == "__main__":
    # export FIGMA_TOKEN=...
    # python figma_visual_diff.py <file_key>
    import sys
    run_visual_poll(sys.argv[1])
