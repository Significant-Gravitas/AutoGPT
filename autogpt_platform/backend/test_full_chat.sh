#!/bin/bash

echo "1. Creating anonymous chat session..."
SESSION_RESPONSE=$(curl -X POST http://localhost:8006/api/v2/chat/sessions \
  -H "Content-Type: application/json" \
  -d '{}' \
  -s)

SESSION_ID=$(echo $SESSION_RESPONSE | python3 -c "import sys, json; print(json.load(sys.stdin)['id'])")
USER_ID=$(echo $SESSION_RESPONSE | python3 -c "import sys, json; print(json.load(sys.stdin)['user_id'])")

echo "✅ Session created:"
echo "   Session ID: $SESSION_ID"
echo "   User ID: $USER_ID"

echo ""
echo "2. Sending message to discover content creation agents..."
echo "   Message: 'I need help with content creation'"
echo ""

curl -X GET "http://localhost:8006/api/v2/chat/sessions/${SESSION_ID}/stream?message=I%20need%20help%20with%20content%20creation&model=gpt-4o&max_context=50" \
  -H "Accept: text/event-stream" \
  --max-time 15 2>&1 | while IFS= read -r line; do
    if [[ $line == data:* ]]; then
        # Extract JSON after "data: "
        json="${line#data: }"
        if [[ $json == *"type"* ]]; then
            type=$(echo "$json" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('type', ''))" 2>/dev/null)
            content=$(echo "$json" | python3 -c "import sys, json; d=json.load(sys.stdin); print(d.get('content', ''))" 2>/dev/null)
            
            if [[ $type == "text" ]]; then
                echo -n "$content"
            elif [[ $type == "html" ]]; then
                if [[ $content == *"Tool:"* ]]; then
                    echo ""
                    echo "🔧 [Tool Call Detected]"
                elif [[ $content == *"auth_required"* ]]; then
                    echo ""
                    echo "🔐 [Authentication Required - Setup blocked for anonymous user]"
                fi
            elif [[ $type == "error" ]]; then
                echo ""
                echo "❌ Error: $content"
            fi
        fi
    fi
done

echo ""
echo ""
echo "✅ Test completed"