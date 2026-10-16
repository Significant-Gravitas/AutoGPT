#!/usr/bin/env bash
# refresh_claude_token.sh — Extract Claude OAuth tokens and update backend/.env
#
# Works on macOS (keychain), Linux (~/.claude/.credentials.json),
# and Windows/WSL (~/.claude/.credentials.json or PowerShell fallback).
#
# Usage:
#   ./scripts/refresh_claude_token.sh              # auto-detect OS
#   ./scripts/refresh_claude_token.sh --env-file /path/to/.env  # custom .env path
#
# Prerequisite: You must have run `claude login` at least once on the host.
#
# The tokens are never printed. The .env file is written owner-only (0600)
# through a temporary file in its own directory, synced and renamed into
# place, so an interrupted run leaves the old file whole. A symbolic link
# or a file another user owns in its place is refused.

set -euo pipefail
umask 077

# --- Parse arguments ---
ENV_FILE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-file) ENV_FILE="$2"; shift 2 ;;
    *) echo "Unknown option: $1"; exit 1 ;;
  esac
done

# Default .env path: relative to this script's location
if [[ -z "$ENV_FILE" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  ENV_FILE="$SCRIPT_DIR/../.env"
fi

# --- Extract tokens by platform ---
ACCESS_TOKEN=""
REFRESH_TOKEN=""

extract_from_credentials_file() {
  local creds_file="$1"
  if [[ -f "$creds_file" ]]; then
    ACCESS_TOKEN=$(jq -r '.claudeAiOauth.accessToken // ""' "$creds_file" 2>/dev/null)
    REFRESH_TOKEN=$(jq -r '.claudeAiOauth.refreshToken // ""' "$creds_file" 2>/dev/null)
  fi
}

case "$(uname -s)" in
  Darwin)
    # macOS: extract from system keychain
    CREDS_JSON=$(security find-generic-password -s "Claude Code-credentials" -w 2>/dev/null || true)
    if [[ -n "$CREDS_JSON" ]]; then
      ACCESS_TOKEN=$(echo "$CREDS_JSON" | jq -r '.claudeAiOauth.accessToken // ""' 2>/dev/null)
      REFRESH_TOKEN=$(echo "$CREDS_JSON" | jq -r '.claudeAiOauth.refreshToken // ""' 2>/dev/null)
    else
      # Fallback to credentials file (e.g. if keychain access denied)
      extract_from_credentials_file "$HOME/.claude/.credentials.json"
    fi
    ;;
  Linux)
    # Linux (including WSL): read from credentials file
    extract_from_credentials_file "$HOME/.claude/.credentials.json"
    ;;
  MINGW*|MSYS*|CYGWIN*)
    # Windows Git Bash / MSYS2 / Cygwin
    APPDATA_PATH="${APPDATA:-$USERPROFILE/AppData/Roaming}"
    extract_from_credentials_file "$APPDATA_PATH/claude/.credentials.json"
    # Fallback to home dir
    if [[ -z "$ACCESS_TOKEN" ]]; then
      extract_from_credentials_file "$HOME/.claude/.credentials.json"
    fi
    ;;
  *)
    echo "Unsupported platform: $(uname -s)"
    exit 1
    ;;
esac

# --- Validate ---
if [[ -z "$ACCESS_TOKEN" ]]; then
  echo "ERROR: Could not extract Claude OAuth token."
  echo ""
  echo "Make sure you have run 'claude login' at least once."
  echo ""
  echo "Locations checked:"
  echo "  macOS:   Keychain ('Claude Code-credentials')"
  echo "  Linux:   ~/.claude/.credentials.json"
  echo "  Windows: %APPDATA%/claude/.credentials.json"
  exit 1
fi

echo "Found the Claude OAuth token."
[[ -n "$REFRESH_TOKEN" ]] && echo "Found the refresh token."

# --- Update .env file ---
if [[ -L "$ENV_FILE" ]]; then
  echo "ERROR: $ENV_FILE is a symbolic link; refusing to write tokens through it." >&2
  exit 1
fi
if [[ -e "$ENV_FILE" && ! -f "$ENV_FILE" ]]; then
  echo "ERROR: $ENV_FILE is not a regular file." >&2
  exit 1
fi
if [[ -e "$ENV_FILE" && ! -O "$ENV_FILE" ]]; then
  echo "ERROR: $ENV_FILE belongs to another user; refusing to write tokens into it." >&2
  exit 1
fi
if [[ ! -e "$ENV_FILE" ]]; then
  echo "WARNING: $ENV_FILE does not exist, creating it."
fi

# Sets each key on its first line (or, with no such line, on its first
# commented-out line), drops repeats and the retired subscription flag, and
# appends keys the file lacks. Values come from the environment, never from
# the command line, where other users could read them.
UPDATE_ENV='
FNR == 1 { pass++ }
pass == 1 {
  if (match($0, /^[A-Za-z_][A-Za-z0-9_]*=/)) active[substr($0, 1, RLENGTH - 1)] = 1
  next
}
BEGIN {
  n = split("CLAUDE_CODE_OAUTH_TOKEN CLAUDE_CODE_REFRESH_TOKEN CHAT_PLATFORM_ROUTE", order, " ")
  value["CLAUDE_CODE_OAUTH_TOKEN"] = ENVIRON["NEW_ACCESS_TOKEN"]
  if (ENVIRON["NEW_REFRESH_TOKEN"] != "") value["CLAUDE_CODE_REFRESH_TOKEN"] = ENVIRON["NEW_REFRESH_TOKEN"]
  value["CHAT_PLATFORM_ROUTE"] = "subscription"
}
{
  # Older versions of this script set this flag; the backend no longer
  # reads it and refuses to start while it is set.
  if ($0 ~ /^CHAT_USE_CLAUDE_CODE_SUBSCRIPTION=/) next
  for (i = 1; i <= n; i++) {
    key = order[i]
    if (!(key in value)) continue
    if (index($0, key "=") == 1 || (!(key in active) && $0 ~ ("^# *" key "="))) {
      if (!(key in done)) { print key "=" value[key]; done[key] = 1 }
      next
    }
  }
  print
}
END {
  for (i = 1; i <= n; i++) {
    key = order[i]
    if ((key in value) && !(key in done)) print key "=" value[key]
  }
}
'

ENV_DIR="$(dirname "$ENV_FILE")"
SOURCE="$ENV_FILE"
[[ -f "$SOURCE" ]] || SOURCE=/dev/null
TMP_FILE="$(mktemp "$ENV_DIR/.env.tmp.XXXXXX")"
trap 'rm -f "$TMP_FILE"' EXIT
chmod 600 "$TMP_FILE"
NEW_ACCESS_TOKEN="$ACCESS_TOKEN" NEW_REFRESH_TOKEN="$REFRESH_TOKEN" \
  awk "$UPDATE_ENV" "$SOURCE" "$SOURCE" > "$TMP_FILE"
# Flush the new file before it replaces the old one (GNU sync takes a file;
# elsewhere, sync everything).
sync "$TMP_FILE" 2>/dev/null || sync
mv -f "$TMP_FILE" "$ENV_FILE"
trap - EXIT
chmod 600 "$ENV_FILE"

echo ""
echo "Updated $ENV_FILE with Claude subscription tokens."
echo "Run 'docker compose up -d copilot_executor' to apply."
