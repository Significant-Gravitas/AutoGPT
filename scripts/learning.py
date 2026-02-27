"""Persistent Learning - Auto-capture insights about the user across sessions.

Every time the agent notices a preference, habit, correction, or workflow
pattern, it writes it here immediately. On the next startup the agent
loads this file and is already smarter.

Storage: auto_gpt_workspace/user_profile.json
"""

import json
import os
import datetime

PROFILE_DIR = os.path.join(os.path.dirname(__file__), '..', 'auto_gpt_workspace')
PROFILE_PATH = os.path.join(PROFILE_DIR, 'user_profile.json')


def _load_profile():
    """Load the user profile from disk, or return a fresh one."""
    os.makedirs(PROFILE_DIR, exist_ok=True)
    if os.path.exists(PROFILE_PATH):
        try:
            with open(PROFILE_PATH, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return _empty_profile()
    return _empty_profile()


def _empty_profile():
    return {
        "preferences": [],
        "workflows": [],
        "corrections": [],
        "facts": [],
    }


def _save_profile(profile):
    """Persist the profile to disk."""
    os.makedirs(PROFILE_DIR, exist_ok=True)
    with open(PROFILE_PATH, 'w') as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)


def learn(category, detail):
    """Record a new insight about the user.

    Args:
        category: One of 'preferences', 'workflows', 'corrections', 'facts'.
        detail: A concise string describing what was learned.

    Returns:
        Confirmation message.
    """
    valid_categories = list(_empty_profile().keys())
    if category not in valid_categories:
        return f"Error: category must be one of {valid_categories}, got '{category}'"

    profile = _load_profile()

    # Avoid exact duplicates
    for entry in profile[category]:
        if entry.get("detail") == detail:
            return f"Already know this: {detail}"

    entry = {
        "detail": detail,
        "recorded": datetime.datetime.now().isoformat(),
    }
    profile[category].append(entry)
    _save_profile(profile)
    return f"Learned ({category}): {detail}"


def recall_learnings():
    """Return everything learned about the user so far.

    Returns:
        A formatted string of all recorded insights, or a notice that
        nothing has been learned yet.
    """
    profile = _load_profile()
    lines = []
    total = 0
    for category, entries in profile.items():
        if entries:
            lines.append(f"\n## {category.upper()}")
            for entry in entries:
                lines.append(f"- {entry['detail']}")
                total += 1

    if total == 0:
        return "No learnings recorded yet."

    return f"User profile ({total} insights):" + "\n".join(lines)


def get_profile_summary():
    """Return a compact summary suitable for injecting into the system prompt.

    Called at startup so the agent begins each session already aware of
    everything it has learned previously.
    """
    profile = _load_profile()
    parts = []
    total = 0
    for category, entries in profile.items():
        if entries:
            items = [e["detail"] for e in entries]
            parts.append(f"{category}: " + "; ".join(items))
            total += len(items)

    if total == 0:
        return ""

    return (
        "WHAT YOU ALREADY KNOW ABOUT THE USER "
        "(loaded from previous sessions):\n"
        + "\n".join(parts)
    )
