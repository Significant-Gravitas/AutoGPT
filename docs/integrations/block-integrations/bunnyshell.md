# Bunnyshell Environment Blocks

## What it is
The Bunnyshell Environment blocks provide programmatic control over [Bunnyshell](https://bunnyshell.com) staging environments. Bunnyshell spins up full Docker Compose stacks — including databases, microservices, and frontends — on demand, with a unique URL per environment.

## What it does
These blocks enable agents to:
- Create isolated staging environments per PR or feature branch
- Deploy your full application stack (10+ microservices, databases, frontends)
- Poll until environments are ready, then run tests against live services
- Stop environments when idle (billing pauses at $0)
- Delete environments when no longer needed

## How it works
Bunnyshell uses a declarative `bunnyshell.yaml` in your repository to define your entire stack. Each environment is fully isolated with its own URLs. Agents interact via Bunnyshell's REST API using a single API token stored in AutoGPT's credential infrastructure.

The typical flow:
1. **Create** an environment from your template
2. **Deploy** to spin up all services
3. **Get Status** — poll until `running`
4. Hit your service URLs directly with `SendWebRequestBlock` for testing
5. **Stop** between sessions (billing pauses)
6. **Delete** when the PR merges (billing stops permanently)

## Prerequisites
- A [Bunnyshell account](https://bunnyshell.com) and API token
- A `bunnyshell.yaml` in your repository defining your stack
- Your Bunnyshell Project ID (found in the dashboard)

## Blocks

### Bunnyshell Create Environment Block

#### What it does
Creates a new Bunnyshell environment from a template. Does not deploy it yet — call `Bunnyshell Deploy Block` next.

#### Inputs
| Input | Description |
|-------|-------------|
| Credentials | Bunnyshell API token |
| Project ID | Your Bunnyshell project ID (from dashboard) |
| Environment Name | Unique name for this environment (e.g. `pr-123`) |
| Template ID | Bunnyshell template ID defining your stack. Leave empty to use repo `bunnyshell.yaml` |
| Labels | Optional key-value labels for tagging environments |

#### Outputs
| Output | Description |
|--------|-------------|
| environment\_id | Unique ID of the created environment — pass to all other blocks |
| environment\_name | Confirmed name of the environment |
| status | Initial status (usually `stopped`) |
| error | Error message if creation failed |

---

### Bunnyshell Deploy Block

#### What it does
Deploys (starts) an existing Bunnyshell environment. This builds Docker images, starts all services, and makes them accessible at their URLs.

#### Inputs
| Input | Description |
|-------|-------------|
| Credentials | Bunnyshell API token |
| Environment ID | ID from `Bunnyshell Create Environment Block` |

#### Outputs
| Output | Description |
|--------|-------------|
| environment\_id | The environment ID (pass through for chaining) |
| status | Deployment status (`deploying`, `running`, `failed`) |
| error | Error message if deployment failed |

---

### Bunnyshell Get Status Block

#### What it does
Checks the current status of a Bunnyshell environment and retrieves all service URLs. Use in a polling loop until status is `running`.

#### Inputs
| Input | Description |
|-------|-------------|
| Credentials | Bunnyshell API token |
| Environment ID | ID of the environment to check |

#### Outputs
| Output | Description |
|--------|-------------|
| status | Current status: `stopped`, `deploying`, `running`, `failed`, `stopping` |
| service\_urls | List of `{name, url}` objects — one per service in your stack |
| is\_ready | Boolean: true when status is `running` |
| error | Error message if status check failed |

---

### Bunnyshell Stop Environment Block

#### What it does
Stops a running environment. All services are shut down but the environment configuration is preserved. **Billing pauses at $0 while stopped.** Use this between PR review sessions.

#### Inputs
| Input | Description |
|-------|-------------|
| Credentials | Bunnyshell API token |
| Environment ID | ID of the environment to stop |

#### Outputs
| Output | Description |
|--------|-------------|
| status | `stopping` or `stopped` |
| environment\_id | The environment ID (pass through) |
| error | Error message if stop failed |

---

### Bunnyshell Delete Environment Block

#### What it does
Permanently deletes a Bunnyshell environment. All services, volumes, and data are removed. Billing stops immediately. Call this when a PR is merged or closed.

#### Inputs
| Input | Description |
|-------|-------------|
| Credentials | Bunnyshell API token |
| Environment ID | ID of the environment to delete |

#### Outputs
| Output | Description |
|--------|-------------|
| status | `deleted` if successful |
| message | Human-readable confirmation |
| error | Error message if deletion failed |

---

## Pricing
Bunnyshell charges **$0.007/minute per active environment**:
- 1 hour active = ~$0.42
- 2 hour PR review session = ~$0.84
- **Stopped environments = $0**

Always stop environments when not actively testing. Delete when the PR is merged.

See [Bunnyshell Pricing](https://bunnyshell.com/pricing) for full details.

## Example Use Case: PR Staging Environment

An agent can automatically create a staging environment for every pull request:
1. **Create** an environment named `pr-{number}`
2. **Deploy** the full stack (backend + databases + frontend)
3. **Get Status** in a polling loop until `is_ready = true` (~3-4 minutes)
4. Use `SendWebRequestBlock` to hit `GET /api/health` — verify all services are up
5. Run integration tests against the live service URLs
6. Post results to the PR comment
7. **Stop** the environment after review (billing pauses)
8. **Delete** the environment when PR merges (billing stops permanently)

Total cost per PR review session: ~$0.84 (2 hours active)

## Combination with E2B Desktop
For full-stack development with live frontend preview, combine with [E2B Desktop Sandbox Blocks](e2b_desktop.md):
- **Bunnyshell** runs your backend microservices + databases (full Docker Compose)
- **E2B Desktop** runs your frontend pointed at the Bunnyshell API URL
- Frontend changes appear in the live stream in ~2 seconds via HMR
- Backend API calls go to the real Bunnyshell staging environment

This gives you a complete full-stack staging environment with instant visual feedback.
