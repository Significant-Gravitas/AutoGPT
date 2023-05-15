## Authentication Endpoints

- `POST /auth/register`: Register a new user
- `POST /auth/login`: Log in an existing user
- `POST /auth/logout`: Log out the current user
- `POST /auth/password-reset`: Initiate a password reset for a user
- `POST /auth/password-reset/confirm`: Confirm the password reset using a provided token

## User Management Endpoints (Admin)

- `GET /admin/users`: Retrieve all users
- `GET /admin/users/{userId}`: Retrieve a specific user
- `POST /admin/users`: Create a new user
- `PUT /admin/users/{userId}`: Update an existing user
- `DELETE /admin/users/{userId}`: Delete a user

## Agent Management (Admin)

- `GET /admin/agents`: Retrieve all agents
- `GET /admin/agents/{agentId}`: Retrieve a specific agent
- `POST /admin/agents`: Create a new agent
- `PUT /admin/agents/{agentId}`: Update an existing agent
- `DELETE /admin/agents/{agentId}`: Delete an agent

## Admin Agent Detailed Information and Metrics (Admin)

- `GET /admin/agents/{agentId}/details`: Retrieve detailed information about a specific agent
- `GET /admin/agents/{agentId}/metrics`: Retrieve metrics about a specific agent
- `GET /admin/agents/{agentId}/logs`: Retrieve logs related to a specific agent
- `GET /admin/agents/{agentId}/costs`: Retrieve resource cost information for a specific agent

## Admin Aggregate Metrics (Admin)

- `GET /admin/metrics`: Retrieve aggregate metrics about all agents
- `GET /admin/costs`: Retrieve aggregate resource cost information for all agents

## User-Agent Interaction Endpoints

- `POST /agents/{agentId}/sessions`: Start a new session with an agent (returns a `sessionId`)
- `GET /agents/{agentId}/sessions/{sessionId}`: Retrieve the current state of a session
- `POST /agents/{agentId}/sessions/{sessionId}/interactions`: Send a message or command to the agent and get a response
- `PUT /agents/{agentId}/sessions/{sessionId}`: Update the state of a session
- `DELETE /agents/{agentId}/sessions/{sessionId}`: End a session

## User's Agents Management

- `GET /users/{userId}/agents`: Retrieve all agents for a specific user
- `GET /users/{userId}/agents/{agentId}`: Retrieve a specific agent for a user
- `POST /users/{userId}/agents`: Create a new agent for a user
- `PUT /users/{userId}/agents/{agentId}`: Update an existing agent for a user
- `DELETE /users/{userId}/agents/{agentId}`: Delete an agent for a user

## User-Agent Detailed Information and Metrics

- `GET /users/{userId}/agents/{agentId}/details`: Retrieve detailed information about a specific agent for a user
- `GET /users/{userId}/agents/{agentId}/metrics`: Retrieve metrics about a specific agent for a user
- `GET /users/{userId}/agents/{agentId}/logs`: Retrieve logs related to a specific agent for a user
- `GET /users/{userId}/agents/{agentId}/costs`: Retrieve resource cost information for a specific agent for a user