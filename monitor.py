#!/usr/bin/env python3
"""Arena.ai leaderboard monitor — detects ranking/score changes and sends ntfy.sh push notifications."""

import json
import os
import sys
from pathlib import Path

import requests
import njsparser

LEADERBOARD_URL = "https://arena.ai/leaderboard/text/overall-no-style-control"
SNAPSHOT_PATH = Path(__file__).parent / "data" / "last_snapshot.json"
TOP_N = 50
SCORE_THRESHOLD = 2.0
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"


def fetch_leaderboard() -> list[dict]:
    """Fetch and parse the arena.ai leaderboard, returning a list of model entries."""
    resp = requests.get(LEADERBOARD_URL, headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    html = resp.text

    entries = parse_with_njsparser(html)
    if not entries:
        entries = parse_with_regex(html)
    if not entries:
        print("ERROR: Could not extract leaderboard data from page", file=sys.stderr)
        sys.exit(1)

    return entries[:TOP_N]


def parse_with_njsparser(html: str) -> list[dict] | None:
    """Primary parser: extract entries from Next.js RSC flight data."""
    try:
        fd = njsparser.BeautifulFD(html)
        if not fd:
            print("njsparser: no flight data found, trying fallback")
            return None

        for data in fd.find_iter([njsparser.Data]):
            if data.content and isinstance(data.content, dict):
                # Look for the leaderboard object containing entries
                if "leaderboard" in data.content:
                    entries = data.content["leaderboard"].get("entries")
                    if entries and isinstance(entries, list):
                        print(f"njsparser: found {len(entries)} entries")
                        return normalize_entries(entries)
                # Also check if entries is directly in content
                if "entries" in data.content:
                    entries = data.content["entries"]
                    if entries and isinstance(entries, list):
                        print(f"njsparser: found {len(entries)} entries (direct)")
                        return normalize_entries(entries)

        print("njsparser: no entries found in flight data, trying fallback")
        return None
    except Exception as e:
        print(f"njsparser: error ({e}), trying fallback")
        return None


def parse_with_regex(html: str) -> list[dict] | None:
    """Fallback parser: regex extraction of model data from raw HTML."""
    import re

    pattern = re.compile(
        r'\{[^{}]*"rank"\s*:\s*(\d+)[^{}]*"modelDisplayName"\s*:\s*"([^"]+)"[^{}]*"rating"\s*:\s*([\d.]+)[^{}]*\}'
    )
    matches = pattern.findall(html)
    if not matches:
        # Try alternate field ordering
        pattern2 = re.compile(
            r'\{[^{}]*"modelDisplayName"\s*:\s*"([^"]+)"[^{}]*"rank"\s*:\s*(\d+)[^{}]*"rating"\s*:\s*([\d.]+)[^{}]*\}'
        )
        matches = [(rank, name, rating) for name, rank, rating in pattern2.findall(html)]

    if not matches:
        return None

    entries = []
    for rank, name, rating in matches:
        entries.append({
            "rank": int(rank),
            "modelDisplayName": name,
            "rating": float(rating),
        })

    entries.sort(key=lambda e: e["rank"])
    # Deduplicate by model name
    seen = set()
    deduped = []
    for e in entries:
        if e["modelDisplayName"] not in seen:
            seen.add(e["modelDisplayName"])
            deduped.append(e)
    print(f"regex fallback: found {len(deduped)} entries")
    return deduped


def normalize_entries(entries: list[dict]) -> list[dict]:
    """Keep only the fields we care about for change detection."""
    fields = ["rank", "modelDisplayName", "rating", "votes", "modelOrganization",
              "license", "rankUpper", "rankLower", "ratingUpper", "ratingLower"]
    normalized = []
    for e in entries:
        normalized.append({k: e.get(k) for k in fields if e.get(k) is not None})
    normalized.sort(key=lambda e: e.get("rank", 999))
    return normalized


def load_snapshot() -> list[dict] | None:
    """Load the previous snapshot from disk."""
    if not SNAPSHOT_PATH.exists():
        return None
    with open(SNAPSHOT_PATH) as f:
        return json.load(f)


def save_snapshot(entries: list[dict]):
    """Save the current snapshot to disk."""
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SNAPSHOT_PATH, "w") as f:
        json.dump(entries, f, indent=2)
    print(f"Snapshot saved ({len(entries)} models)")


def detect_changes(old: list[dict], new: list[dict]) -> dict:
    """Compare two snapshots and return a dict of changes."""
    old_by_name = {e["modelDisplayName"]: e for e in old}
    new_by_name = {e["modelDisplayName"]: e for e in new}

    rank_changes = []
    score_changes = []
    new_models = []
    removed_models = []

    for name, entry in new_by_name.items():
        if name not in old_by_name:
            new_models.append(entry)
            continue
        old_entry = old_by_name[name]
        # Rank change
        if entry.get("rank") != old_entry.get("rank"):
            rank_changes.append({
                "model": name,
                "old_rank": old_entry.get("rank"),
                "new_rank": entry.get("rank"),
                "rating": entry.get("rating"),
            })
        # Score change
        old_rating = old_entry.get("rating", 0)
        new_rating = entry.get("rating", 0)
        if old_rating and new_rating and abs(new_rating - old_rating) >= SCORE_THRESHOLD:
            score_changes.append({
                "model": name,
                "old_rating": old_rating,
                "new_rating": new_rating,
                "diff": round(new_rating - old_rating, 2),
            })

    for name, entry in old_by_name.items():
        if name not in new_by_name:
            removed_models.append(entry)

    return {
        "rank_changes": rank_changes,
        "score_changes": score_changes,
        "new_models": new_models,
        "removed_models": removed_models,
    }


def has_changes(changes: dict) -> bool:
    return any(changes[k] for k in changes)


def format_notification(changes: dict) -> tuple[str, str]:
    """Format changes into a notification title and body."""
    parts = []
    counts = []

    if changes["rank_changes"]:
        counts.append(f"{len(changes['rank_changes'])} rank")
        parts.append("Rank changes:")
        for c in sorted(changes["rank_changes"], key=lambda x: x["new_rank"]):
            direction = "^" if c["new_rank"] < c["old_rank"] else "v"
            parts.append(f"  {direction} {c['model']}: #{c['old_rank']} -> #{c['new_rank']} ({c['rating']:.2f})")

    if changes["score_changes"]:
        counts.append(f"{len(changes['score_changes'])} score")
        if parts:
            parts.append("")
        parts.append("Score changes:")
        for c in sorted(changes["score_changes"], key=lambda x: abs(x["diff"]), reverse=True):
            sign = "+" if c["diff"] > 0 else ""
            parts.append(f"  {c['model']}: {c['old_rating']:.2f} -> {c['new_rating']:.2f} ({sign}{c['diff']:.2f})")

    if changes["new_models"]:
        counts.append(f"{len(changes['new_models'])} new")
        if parts:
            parts.append("")
        parts.append("New models:")
        for e in changes["new_models"]:
            parts.append(f"  + {e['modelDisplayName']} (#{e.get('rank', '?')}, {e.get('rating', 0):.2f})")

    if changes["removed_models"]:
        counts.append(f"{len(changes['removed_models'])} removed")
        if parts:
            parts.append("")
        parts.append("Removed from top {TOP_N}:")
        for e in changes["removed_models"]:
            parts.append(f"  - {e['modelDisplayName']} (was #{e.get('rank', '?')})")

    title = f"Arena Leaderboard: {', '.join(counts)} change(s)"
    body = "\n".join(parts)
    return title, body


def involves_top5(changes: dict) -> bool:
    """Check if any change involves a top-5 model."""
    for c in changes["rank_changes"]:
        if c["new_rank"] <= 5 or c["old_rank"] <= 5:
            return True
    for c in changes["score_changes"]:
        # Check score changes for top-5 models by looking at rank_changes too
        pass
    for e in changes["new_models"]:
        if e.get("rank", 999) <= 5:
            return True
    return False


def send_notification(title: str, body: str, priority: int = 3):
    """Send a push notification via ntfy.sh."""
    topic = os.environ.get("NTFY_TOPIC")
    if not topic:
        print("NTFY_TOPIC not set, skipping notification")
        return

    try:
        resp = requests.post(
            f"https://ntfy.sh/{topic}",
            data=body.encode("utf-8"),
            headers={
                "Title": title,
                "Priority": str(priority),
                "Click": LEADERBOARD_URL,
                "Tags": "chart_with_upwards_trend",
            },
            timeout=10,
        )
        resp.raise_for_status()
        print(f"Notification sent (priority {priority})")
    except Exception as e:
        print(f"WARNING: Failed to send notification: {e}", file=sys.stderr)


def main():
    print(f"Fetching leaderboard from {LEADERBOARD_URL}")
    current = fetch_leaderboard()
    print(f"Fetched {len(current)} models, top model: {current[0]['modelDisplayName']} (#{current[0]['rank']})")

    previous = load_snapshot()

    if previous is None:
        # First run — save baseline
        save_snapshot(current)
        top3 = ", ".join(f"#{e['rank']} {e['modelDisplayName']}" for e in current[:3])
        send_notification(
            "Arena Monitor Started",
            f"Tracking top {TOP_N} models on arena.ai\nCurrent top 3: {top3}",
            priority=2,
        )
        print("First run — baseline saved")
        return

    changes = detect_changes(previous, current)

    if not has_changes(changes):
        print("No meaningful changes detected")
        save_snapshot(current)
        return

    title, body = format_notification(changes)
    priority = 4 if involves_top5(changes) else 3
    print(f"\n{title}\n{body}\n")

    send_notification(title, body, priority)
    save_snapshot(current)


if __name__ == "__main__":
    main()
