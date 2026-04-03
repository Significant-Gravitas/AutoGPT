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

set -euo pipefail

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

echo "Found Claude OAuth token: ${ACCESS_TOKEN:0:20}..."
[[ -n "$REFRESH_TOKEN" ]] && echo "Found refresh token:  ${REFRESH_TOKEN:0:20}..."

# --- Update .env file ---
update_env_var() {
  local key="$1" value="$2" file="$3"
  if grep -q "^${key}=" "$file" 2>/dev/null; then
    # Replace existing value (works on both macOS and Linux sed)
    if [[ "$(uname -s)" == "Darwin" ]]; then
      sed -i '' "s|^${key}=.*|${key}=${value}|" "$file"
    else
      sed -i "s|^${key}=.*|${key}=${value}|" "$file"
    fi
  elif grep -q "^# *${key}=" "$file" 2>/dev/null; then
    # Uncomment and set
    if [[ "$(uname -s)" == "Darwin" ]]; then
      sed -i '' "s|^# *${key}=.*|${key}=${value}|" "$file"
    else
      sed -i "s|^# *${key}=.*|${key}=${value}|" "$file"
    fi
  else
    # Append
    echo "${key}=${value}" >> "$file"
  fi
}

if [[ ! -f "$ENV_FILE" ]]; then
  echo "WARNING: $ENV_FILE does not exist, creating it."
  touch "$ENV_FILE"
fi

update_env_var "CLAUDE_CODE_OAUTH_TOKEN" "$ACCESS_TOKEN" "$ENV_FILE"
[[ -n "$REFRESH_TOKEN" ]] && update_env_var "CLAUDE_CODE_REFRESH_TOKEN" "$REFRESH_TOKEN" "$ENV_FILE"
update_env_var "CHAT_USE_CLAUDE_CODE_SUBSCRIPTION" "true" "$ENV_FILE"

echo ""
echo "Updated $ENV_FILE with Claude subscription tokens."
echo "Run 'docker compose up -d copilot_executor' to apply."
