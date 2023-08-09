from dotenv import load_dotenv

import autogpt.agent

if __name__ == "__main__":
    """Runs the agent server"""
    load_dotenv()
    autogpt.agent.start_agent(port=8915)

