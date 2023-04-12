from llm_utils import create_chat_completion

next_key = {None: 0}
agents = {None: {}}  # key, (task, full_message_history, model)

# Create new GPT agent
# TODO: Centralise use of create_chat_completion() to globally enforce token limit


def create_agent(task, prompt, model, chat_id=None):
    """Create a new agent and return its key"""
    global next_key
    global agents

    messages = [{"role": "user", "content": prompt}, ]

    # Start GPT instance
    agent_reply = create_chat_completion(
        model=model,
        messages=messages,
    )

    # Update full message history
    messages.append({"role": "assistant", "content": agent_reply})

    if not chat_id in next_key:
        next_key[chat_id] = 0
    key = next_key[chat_id]
    # This is done instead of len(agents) to make keys unique even if agents
    # are deleted
    next_key[chat_id] += 1

    if not chat_id in agents:
        agents[chat_id] = {}
    agents[chat_id][key] = (task, messages, model)

    return key, agent_reply


def message_agent(key, message, chat_id=None):
    """Send a message to an agent and return its response"""
    global agents

    task, messages, model = agents[chat_id][int(key)]

    # Add user message to message history before sending to agent
    messages.append({"role": "user", "content": message})

    # Start GPT instance
    agent_reply = create_chat_completion(
        model=model,
        messages=messages,
    )

    # Update full message history
    messages.append({"role": "assistant", "content": agent_reply})
    agents[chat_id][int(key)] = (task, messages, model)

    return agent_reply


def list_agents(chat_id=None):
    """Return a list of all agents"""
    global agents

    # Return a list of agent keys and their tasks
    return [(key, task) for key, (task, _, _) in agents[chat_id].items()]


def delete_agent(key, chat_id=None):
    """Delete an agent and return True if successful, False otherwise"""
    global agents

    try:
        del agents[chat_id][int(key)]
        return True
    except KeyError:
        return False
