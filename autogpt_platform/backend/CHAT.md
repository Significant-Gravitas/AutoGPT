# Chat / Discover Flow

- Open discover page

if session id load old session

else:

- user sends message
- ai responds and does tool calls to find agents

- find agent widget is displayed.

If the user selects and agent or asks the ai for more details

- find agent details is called.

This starts a complex flow.

1) Firstly if they are not logged in they are asked to login
2) Then agent details are retrived

If they click setup or ask for it to be setup then we 

4) We then check to see if the user has credentials setup for the required credentials if not we add a ui element for each credential that needs setting up.
5) Ask the user for the required input data.
6) suggest / get told what schedule to run it on if it is not a triggered agent 
7) sets up the agent. (adds it to there library and installs the trigger or schedules the agent)
8) confirms the agent has been setup with a ui that navigates them to the setup agent

END

