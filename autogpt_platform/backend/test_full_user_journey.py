#!/usr/bin/env python3
"""
Full User Journey Test Script for AutoGPT Chat-Based Discovery

This script uses OpenAI to simulate a realistic user who needs leads for their business:
1. Start anonymous chat session
2. User expresses business need for lead generation
3. AI simulates natural conversation flow
4. Handle authentication when prompted
5. Get agent details and setup
6. Complete the journey dynamically

The AI user will:
- Start with a business problem (need leads for their startup)
- Ask natural questions
- Respond appropriately to tool calls and agent suggestions
- Handle authentication flow
- Set up the chosen agent
"""

import json
import os
import sys
import time

import requests

try:
    from openai import OpenAI

    openai_available = True
except ImportError:
    openai_available = False
    print("⚠️ OpenAI not available, falling back to static messages")


class ChatTestClient:
    def __init__(self, base_url: str = "http://localhost:8006"):
        self.base_url = base_url
        self.session_id = None
        self.user_id = None
        self.auth_token = None
        self.conversation_history = []
        self.tool_calls_detected = []
        self.openai_client = (
            OpenAI(api_key=os.getenv("OPENAI_API_KEY")) if openai_available else None
        )

    def log(self, message: str, level: str = "INFO"):
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")

    def generate_user_message(self, context: str = "", is_initial: bool = False) -> str:
        """Generate a realistic user message using OpenAI or fallback to static messages"""
        if not self.openai_client:
            # Fallback to static messages
            return self._get_fallback_message(context, is_initial)

        try:
            # Build conversation context for AI user
            system_prompt = """You are a business owner named Sarah who runs a B2B SaaS startup called "TechFlow Solutions".
You provide workflow automation tools for small businesses. You need to find new leads/customers for your business.

Your personality:
- Friendly and professional
- Knowledgeable about B2B sales
- Looking for qualified leads in your target market
- Open to automation tools that can help your sales process
- Willing to try new technology solutions

Current conversation context: {context}

Generate a natural, conversational response that continues this discussion. Keep it realistic - ask questions, express interest, or take next steps based on what the AI assistant has said.

Respond as Sarah would in a real conversation about finding leads for her business.
Keep response super short and concise.
"""

            user_context = ""
            if self.conversation_history:
                user_context = "\n".join(
                    [
                        f"{'Sarah' if i % 2 == 0 else 'Assistant'}: {msg}"
                        for i, msg in enumerate(self.conversation_history[-4:])
                    ]
                )

            if is_initial:
                user_prompt = "Start the conversation by introducing yourself and explaining that you're looking for ways to find new leads for your B2B SaaS startup. Be natural and conversational."
            else:
                user_prompt = f"Continue the conversation naturally. Previous messages:\n{user_context}\n\nRespond as Sarah would."

            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt.format(context=context),
                    },
                    {"role": "user", "content": user_prompt},
                ],
                max_completion_tokens=150,
                temperature=0.8,
            )

            message = response.choices[0].message.content.strip()
            self.log(f"🤖 AI User: {message}", "USER")
            return message

        except Exception as e:
            self.log(f"AI generation failed: {e}, using fallback", "WARN")
            return self._get_fallback_message(context, is_initial)

    def _get_fallback_message(self, context: str, is_initial: bool) -> str:
        """Fallback static messages when OpenAI is not available"""
        if is_initial:
            return "Hi! I'm Sarah from TechFlow Solutions, a B2B SaaS company. I'm looking for ways to find more qualified leads for my workflow automation tools. Do you have any agents that could help with lead generation and prospecting?"

        if "auth_required" in context.lower():
            return "I understand I need to sign in. Let me do that now."

        if "agent" in context.lower() and "found" in context.lower():
            return "These agents look interesting! Can you tell me more about the lead generation one? What does it do exactly?"

        if "setup" in context.lower():
            return "That sounds perfect! Can you help me set up this lead generation agent to run every day at 9 AM?"

        return "Thanks for the information! What else can you help me with?"

    def create_session(self) -> bool:
        """Create an anonymous chat session"""
        try:
            self.log("Creating anonymous chat session...")
            response = requests.post(
                f"{self.base_url}/api/v2/chat/sessions",
                json={},
                headers={"Content-Type": "application/json"},
            )

            if response.status_code == 200:
                data = response.json()
                self.session_id = data["id"]
                self.user_id = data["user_id"]
                self.log(f"✅ Session created: {self.session_id}")
                self.log(f"   User ID: {self.user_id}")
                return True
            else:
                self.log(
                    f"❌ Failed to create session: {response.status_code} - {response.text}",
                    "ERROR",
                )
                return False

        except Exception as e:
            self.log(f"❌ Error creating session: {e}", "ERROR")
            return False

    def send_message(self, message: str = None, context: str = "") -> str:
        """Send a message and return the response using streaming endpoint.

        Always uses the /sessions/{session_id}/stream endpoint for consistency,
        except when authentication is required (handled separately).
        """
        if not self.session_id:
            self.log("❌ No session ID available", "ERROR")
            return ""

        # Generate AI-powered message if none provided
        if message is None:
            is_initial = len(self.conversation_history) == 0
            message = self.generate_user_message(context, is_initial)

        # Add user message to conversation history
        self.conversation_history.append(message)

        try:
            self.log(f"💬 Sending message via stream: {message[:100]}...")

            # Use streaming endpoint for real-time response
            params = {"message": message, "model": "gpt-4o", "max_context": 50}

            response = requests.get(
                f"{self.base_url}/api/v2/chat/sessions/{self.session_id}/stream",
                params=params,
                headers={"Accept": "text/event-stream"},
                stream=True,
            )

            if response.status_code != 200:
                self.log(f"❌ Stream request failed: {response.status_code}", "ERROR")
                return ""

            # Process SSE stream
            full_response = ""
            tool_calls = []
            auth_required = False
            agent_results = []

            for line in response.iter_lines():
                if line:
                    line = line.decode("utf-8")
                    if line.startswith("data: "):
                        data = line[6:].strip()

                        if data == "[DONE]":
                            break

                        try:
                            chunk = json.loads(data)
                            chunk_type = chunk.get("type")
                            content = chunk.get("content", "")

                            # Check for tool calls in the chunk
                            if "tool_calls" in chunk and chunk["tool_calls"]:
                                tool_calls_info = chunk["tool_calls"]
                                if isinstance(tool_calls_info, list):
                                    for tool_call in tool_calls_info:
                                        if isinstance(tool_call, dict):
                                            tool_name = tool_call.get(
                                                "function", {}
                                            ).get("name", "unknown")
                                            self.log(
                                                f"🔧 Tool Call: {tool_name}", "TOOL"
                                            )
                                            tool_call_info = {
                                                "name": tool_name,
                                                "arguments": tool_call.get(
                                                    "function", {}
                                                ).get("arguments", ""),
                                                "id": tool_call.get("id", ""),
                                            }
                                            tool_calls.append(tool_call_info)
                                            self.tool_calls_detected.append(
                                                tool_call_info
                                            )

                            if chunk_type == "text":
                                full_response += content
                                # Don't print each chunk separately as it adds newlines
                            elif chunk_type == "html":
                                # Parse HTML for tool calls and special content
                                if "auth_required" in content:
                                    auth_required = True
                                    self.log(
                                        "🔐 Authentication required detected", "AUTH"
                                    )
                                elif "agents matching" in content:
                                    self.log(
                                        "🤖 Agent search results detected", "AGENT"
                                    )
                                    # Extract agent data from HTML
                                    try:
                                        # Look for JSON in the HTML content
                                        import re

                                        json_match = re.search(r"\[.*?\]", content)
                                        if json_match:
                                            agents = json.loads(json_match.group())
                                            agent_results.extend(agents)
                                    except Exception:
                                        pass
                                # Check for tool call indicators in HTML
                                elif "Calling Tool:" in content:
                                    self.log(
                                        "🔧 Tool call execution detected in HTML",
                                        "TOOL",
                                    )
                                elif "tool-call-container" in content:
                                    self.log("🔧 Tool call UI element detected", "TOOL")

                        except json.JSONDecodeError:
                            continue

            # Print the full response after streaming completes
            if full_response:
                print(f"\n💬 Assistant: {full_response}")
            self.log(f"📝 Full response length: {len(full_response)} characters")

            # Log tool calls summary
            if tool_calls:
                self.log(f"🔧 Tool Calls Made: {len(tool_calls)}", "TOOL")
                for tool_call in tool_calls:
                    self.log(f"   - {tool_call['name']}", "TOOL")

            # Add assistant response to conversation history
            if full_response:
                self.conversation_history.append(full_response)

            # Check for special conditions
            if auth_required:
                self.log("🔐 AUTHENTICATION REQUIRED - User needs to sign in", "AUTH")

            if agent_results:
                self.log(f"🤖 Found {len(agent_results)} agents", "AGENT")
                for agent in agent_results[:3]:  # Show first 3
                    self.log(
                        f"   - {agent.get('name', 'Unknown')}: {agent.get('description', 'No description')[:100]}...",
                        "AGENT",
                    )

            return full_response

        except Exception as e:
            self.log(f"❌ Error sending message: {e}", "ERROR")
            return ""

    def simulate_auth(self) -> bool:
        """Simulate user authentication and claim the session"""
        if not self.session_id:
            self.log("❌ No session ID available", "ERROR")
            return False

        try:
            self.log("🔐 Simulating user authentication...")

            # NOTE: For real authentication, you would need:
            # 1. A valid Supabase JWT token from actual login
            # 2. Or configure test authentication in the backend
            # For now, we'll skip the actual API call and just simulate success

            self.log("⚠️ Skipping actual authentication (requires valid JWT)", "AUTH")
            self.log(
                "📝 In production, user would login via Supabase and receive JWT",
                "AUTH",
            )

            # Simulate successful authentication
            mock_user_id = "test_user_123"
            self.user_id = mock_user_id
            self.log(f"✅ Simulated authentication for user {mock_user_id}", "AUTH")

            return True

        except Exception as e:
            self.log(f"❌ Error during authentication: {e}", "ERROR")
            return False

    def setup_agent(self, agent_id: str, agent_name: str) -> bool:
        """Set up an agent for daily execution"""
        try:
            self.log(
                f"⚙️ Setting up agent {agent_name} ({agent_id}) for daily execution..."
            )

            # This would use the setup_agent tool through the chat
            setup_message = f"Set up the agent '{agent_name}' (ID: {agent_id}) to run every day at 9 AM for lead generation"

            response = self.send_message(setup_message)

            if response:
                self.log("✅ Agent setup request sent successfully", "SETUP")
                return True
            else:
                self.log("❌ Failed to send agent setup request", "ERROR")
                return False

        except Exception as e:
            self.log(f"❌ Error setting up agent: {e}", "ERROR")
            return False


def run_dynamic_journey():
    """Run the complete user journey test with AI-powered dynamic conversation"""
    client = ChatTestClient()

    print("🚀 Starting Dynamic AI User Journey Test")
    print("=" * 60)
    print(
        "🤖 Sarah (AI User): Business owner looking for leads for her B2B SaaS startup"
    )
    print("=" * 60)

    # Step 1: Create anonymous session
    if not client.create_session():
        print("❌ Test failed at session creation")
        return False

    print("\n" + "=" * 60)

    # Step 2: Dynamic conversation flow using streaming endpoint
    print("💬 Step 2: Dynamic AI-powered conversation (using streaming endpoint)...")

    max_turns = 8  # Limit conversation turns to prevent infinite loops
    turn_count = 0
    auth_handled = False

    while turn_count < max_turns:
        turn_count += 1
        print(f"\n🔄 Turn {turn_count}/{max_turns} (streaming)")

        # Send AI-generated user message via streaming endpoint
        context = f"Turn {turn_count}. Auth handled: {auth_handled}"
        response = client.send_message(context=context)

        if not response:
            print(f"❌ Test failed at turn {turn_count}")
            return False

        # Check if authentication is required and handle it
        if not auth_handled and (
            "auth_required" in response.lower() or "sign in" in response.lower()
        ):
            print("\n🔐 Authentication detected - user needs to sign in...")

            # Simulate user signing in
            if not client.simulate_auth():
                print("❌ Authentication failed")
                return False
            auth_handled = True
            print("✅ Authentication complete - user is now signed in")

            # After authentication, continue with the same streaming endpoint
            # The next message will be handled with the authenticated user

        # Check if agent setup is happening
        if "setup" in response.lower() and "agent" in response.lower():
            print("\n⚙️ Agent setup initiated via streaming!")

        # Check for completion indicators
        if any(
            keyword in response.lower()
            for keyword in [
                "completed",
                "successfully set up",
                "ready to run",
                "all done",
            ]
        ):
            print("🎉 Journey appears complete!")
            break

        # Small delay between turns
        time.sleep(1)

    print("\n" + "=" * 60)
    print("🎉 Dynamic AI User Journey Test Completed!")
    print("✅ Conversation flow:")
    print(f"   - {turn_count} conversation turns completed")
    print("   - Anonymous session created")
    print("   - Dynamic AI user responses generated")
    print(f"   - Authentication handled: {auth_handled}")
    print(f"   - Tool calls detected: {len(client.tool_calls_detected)}")
    print("   - Agent discovery and setup attempted")

    # Show tool calls summary
    if client.tool_calls_detected:
        print("\n🔧 Tool Calls Summary:")
        tool_call_counts = {}
        for tool_call in client.tool_calls_detected:
            tool_name = tool_call["name"]
            tool_call_counts[tool_name] = tool_call_counts.get(tool_name, 0) + 1

        for tool_name, count in tool_call_counts.items():
            print(f"   - {tool_name}: {count} call{'s' if count > 1 else ''}")

    # Show conversation summary
    print("\n📝 Conversation Summary:")
    for i, msg in enumerate(client.conversation_history):
        role = "🤖 Sarah" if i % 2 == 0 else "🤖 Assistant"
        print(f"   {role}: {msg[:80]}{'...' if len(msg) > 80 else ''}")

    return True


if __name__ == "__main__":
    print("AutoGPT Chat-Based Discovery - Dynamic AI User Journey Test")
    print("=" * 60)

    # Check OpenAI availability
    if openai_available and os.getenv("OPENAI_API_KEY"):
        print("✅ OpenAI available - AI-powered user simulation enabled")
    else:
        print("⚠️ OpenAI not available - using fallback static messages")

    try:
        success = run_dynamic_journey()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⏹️ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Test failed with unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
