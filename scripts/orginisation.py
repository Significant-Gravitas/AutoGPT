from typing import List

from agent import Agent
from colorama import Fore, Style
from helper import print_to_console


# An orginization conistis of multiple agent.
class Orginization():
    def __init__(self, name):
        self.orginization_name = name # The name of the org
        self.agents: List[Agent] = [] # All the active agents in the orginization


    def run(self):
        # for all agents in agents run the agent.run functiony
        while True:
            for agent in self.agents:
                # Print what agen wey are running
                print(Fore.GREEN + f"\n ---------------- Running agent {agent.agent_id} -----------------")
                agent.step()
                print ("\n")
    
    
    def create_agent(self, name=None, task=None, prompt=None, supervisor_id=None, supervisor_name=None, 
                     founder=False):
        # print the task and supervisor
        print(f"ORG: Creating agent with task {task} and supervisor {supervisor_name}")
        new_agent = Agent(agent_id=len(self.agents),
                          orginization=self,
                          agent_name=name,
                          task=task,
                          goals=prompt,
                          supervisor_id=supervisor_id,
                          supervisor_name=supervisor_name,
                          founder=founder)
        self.agents.append(new_agent)
        print("ORG: orginization agents = ", self.agents)
        return new_agent # return the agent to the supervisor


    def remove_agent(self, agent):
        self.agents.remove(agent)


    def route_message(self, sender, reciever, message):
        print_to_console("ORG: Route message",
                         Fore.RED,
                         f'ORG: Sender: {sender.agent_id} \n ORG: Reciever: {reciever.agent_id} \n ORG: Message: {message} \n')
        reciever.recieve_message(sender, message)

if __name__ == "__main__":
    org = Orginization("A cool AGI orginization")
    org.create_agent(founder=True) # Founder of te orginization.
    org.run() # Run te org


        