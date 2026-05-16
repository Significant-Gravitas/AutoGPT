#!/bin/bash
set -e

# CRE Analyzer startup script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Check for ANTHROPIC_API_KEY
if [ -z "$ANTHROPIC_API_KEY" ]; then
  if [ -f "$SCRIPT_DIR/backend/.env" ]; then
    export $(cat "$SCRIPT_DIR/backend/.env" | xargs)
  else
    echo "⚠️  Warning: ANTHROPIC_API_KEY not set. Document parsing will fail."
    echo "   Copy backend/.env.example to backend/.env and add your key."
  fi
fi

# Add node to PATH if via homebrew
export PATH="/opt/homebrew/bin:$PATH"

echo "🏗️  Starting CRE Deal Analyzer..."

# Start backend
echo "  → Starting FastAPI backend on :8000"
cd "$SCRIPT_DIR/backend"
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
  .venv/bin/pip install -q -r requirements.txt
fi
.venv/bin/python main.py &
BACKEND_PID=$!

# Start frontend
echo "  → Starting Vite frontend on :5173"
cd "$SCRIPT_DIR/frontend"
if [ ! -d "node_modules" ]; then
  npm install
fi
npm run dev &
FRONTEND_PID=$!

echo ""
echo "✅ CRE Deal Analyzer running!"
echo "   Frontend: http://localhost:5173"
echo "   Backend:  http://localhost:8000"
echo "   API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop both servers."

# Wait and cleanup
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; echo 'Stopped.'" SIGINT SIGTERM
wait
