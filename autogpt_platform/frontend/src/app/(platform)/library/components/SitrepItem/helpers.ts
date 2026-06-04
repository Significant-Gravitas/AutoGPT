import { SitrepItemData } from "../../types";

export function buildAutoPilotPrompt(item: SitrepItemData): string {
  switch (item.priority) {
    case "error":
      return `What happened with ${item.agentName}? It says "${item.message}" — can you check the logs and tell me what to fix?`;
    case "running":
      return `Give me a status update on the ${item.agentName} run — what has it found so far?`;
    case "stale":
      return `${item.agentName} hasn't run recently. Should I keep it or update and re-run it?`;
    case "success":
      return `Show me what ${item.agentName} found in its last run — summarize the results and any key takeaways.`;
    case "listening":
      return `What is ${item.agentName} listening for? Give me a summary of its trigger configuration.`;
    case "scheduled":
      return `When is ${item.agentName} scheduled to run next?`;
    case "idle":
      return `${item.agentName} has been idle. Should I keep it or update and re-run it?`;
  }
}
