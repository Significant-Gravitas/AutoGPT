from autogpt.llm_utils import create_chat_completion

next_key = 0
agents = {}  # key, (task, full_message_history, model)

# Create new GPT agent
# TODO: Centralise use of create_chat_completion() to globally enforce token limit


def create_agent(task, prompt, model):
    """Create a new agent and return its key"""
    global next_key
    global agents

    messages = [
        {"role": "user", "content": prompt},
    ]

    # Start GPT instance
    agent_reply = create_chat_completion(
        model=model,
        messages=messages,
    )

    # Update full message history
    messages.append({"role": "assistant", "content": agent_reply})

    key = next_key
    # This is done instead of len(agents) to make keys unique even if agents
    # are deleted
    next_key += 1

    agents[key] = (task, messages, model)

    return key, agent_reply


def message_agent(key, message):
    """Send a message to an agent and return its response"""
    global agents

    task, messages, model = agents[int(key)]

    # Add user message to message history before sending to agent
    messages.append({"role": "user", "content": message})

    # Start GPT instance
    agent_reply = create_chat_completion(
        model=model,
        messages=messages,
    )

    # Update full message history
    messages.append({"role": "assistant", "content": agent_reply})

    return agent_reply


def list_agents():
    """Return a list of all agents"""
    global agents

    # Return a list of agent keys and their tasks
    return [(key, task) for key, (task, _, _) in agents.items()]


def delete_agent(key):
    """Delete an agent and return True if successful, False otherwise"""
    global agents

    try:
        del agents[int(key)]
        return True
    except KeyError:
        return False
