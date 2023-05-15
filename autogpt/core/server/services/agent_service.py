async def get_all_agents():
    return {"message": "get_all_agents has been run"}


async def get_agent(agent_id):
    return {"message": f"get_agent has been run with agentId: {agent_id}"}


async def create_agent():
    return {"message": "create_agent has been run"}


async def update_agent(agent_id):
    return {"message": f"update_agent has been run with agentId: {agent_id}"}


async def delete_agent(agent_id):
    return {"message": f"delete_agent has been run with agentId: {agent_id}"}


async def get_agent_details(agent_id):
    return {"message": f"get_agent_details has been run with agentId: {agent_id}"}


async def get_agent_metrics(agent_id):
    return {"message": f"get_agent_metrics has been run with agentId: {agent_id}"}


async def get_agent_logs(agent_id):
    return {"message": f"get_agent_logs has been run with agentId: {agent_id}"}


async def get_agent_costs(agent_id):
    return {"message": f"get_agent_costs has been run with agentId: {agent_id}"}
