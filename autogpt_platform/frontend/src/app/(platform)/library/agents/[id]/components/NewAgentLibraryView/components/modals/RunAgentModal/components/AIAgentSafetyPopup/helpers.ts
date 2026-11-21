import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { Key, storage } from "@/services/storage/local-storage";

// A step only waits for review while its safe mode is on. Missing settings
// fall back to the backend's GraphSettings defaults: human-in-the-loop review
// on, sensitive action review off.
export function getSafetyChecks(agent: LibraryAgent) {
  return {
    humanInTheLoop:
      agent.has_human_in_the_loop &&
      (agent.settings?.human_in_the_loop_safe_mode ?? true),
    sensitiveActions:
      agent.has_sensitive_action &&
      (agent.settings?.sensitive_action_safe_mode ?? false),
  };
}

export function shouldShowSafetyPopup(agent: LibraryAgent) {
  const checks = getSafetyChecks(agent);
  const hasCheckOn = checks.humanInTheLoop || checks.sensitiveActions;
  return hasCheckOn && !getSeenAgentIDs().includes(agent.id);
}

export function markSafetyPopupSeen(agentID: string) {
  const seenAgentIDs = getSeenAgentIDs();
  if (seenAgentIDs.includes(agentID)) return;
  storage.set(
    Key.AI_AGENT_SAFETY_POPUP_SHOWN,
    JSON.stringify([...seenAgentIDs, agentID]),
  );
}

// A stored value that isn't a list of IDs counts as none seen, so the run
// still starts.
function getSeenAgentIDs(): string[] {
  const seenAgentsJSON = storage.get(Key.AI_AGENT_SAFETY_POPUP_SHOWN);
  if (!seenAgentsJSON) return [];
  try {
    const seen: unknown = JSON.parse(seenAgentsJSON);
    return Array.isArray(seen)
      ? seen.filter((id): id is string => typeof id === "string")
      : [];
  } catch {
    return [];
  }
}
