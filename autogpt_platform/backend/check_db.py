import asyncio
from prisma import Prisma

async def main():
    db = Prisma()
    await db.connect()
    
    # Check existing data
    user_count = await db.user.count()
    agent_count = await db.agentgraph.count()
    library_count = await db.libraryagent.count()
    
    print(f"Existing data in database:")
    print(f"- Users: {user_count}")
    print(f"- Agent Graphs: {agent_count}")
    print(f"- Library Agents: {library_count}")
    
    if user_count > 0 or agent_count > 0 or library_count > 0:
        print("\nDatabase already contains data!")
        print("You may want to clean it before running test data creator.")
    
    await db.disconnect()

if __name__ == "__main__":
    asyncio.run(main())