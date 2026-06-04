import { SitrepItemData } from "../../types";

export function buildAutoPilotPrompt(item: SitrepItemData): string {
  // Embed the exact library agent ID and steer the assistant to resolve it by
  // ID (find_library_agent does a direct lookup when given a UUID) instead of a
  // fuzzy name search.
  const idHint = ` Use find_library_agent with the exact ID ${item.agentID} to look it up.`;
  switch (item.priority) {
    case "error":
      return `What happened with ${item.agentName}? It says "${item.message}" — can you check the logs and tell me what to fix?${idHint}`;
    case "running":
      return `Give me a status update on the ${item.agentName} run — what has it found so far?${idHint}`;
    case "stale":
      return `${item.agentName} hasn't run recently. Should I keep it or update and re-run it?${idHint}`;
    case "success":
      return `Show me what ${item.agentName} found in its last run — summarize the results and any key takeaways.${idHint}`;
    case "listening":
      return `What is ${item.agentName} listening for? Give me a summary of its trigger configuration.${idHint}`;
    case "scheduled":
      return `When is ${item.agentName} scheduled to run next?${idHint}`;
    case "idle":
      return `${item.agentName} has been idle. Should I keep it or update and re-run it?${idHint}`;
  }
}
