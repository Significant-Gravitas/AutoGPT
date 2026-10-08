export type AgentStatus = "idle" | "thinking" | "working" | "waiting" | "done";

export const AGENT_STATUS_LABEL: Record<AgentStatus, string> = {
  idle: "Ready",
  thinking: "Thinking",
  working: "Working",
  waiting: "Needs you",
  done: "Done",
};

export const AGENT_STATUS_BADGE_CLASS: Record<AgentStatus, string> = {
  idle: "bg-zinc-300",
  thinking: "bg-purple-400 animate-pulse motion-reduce:animate-none",
  working: "bg-purple-500 animate-pulse motion-reduce:animate-none",
  waiting: "bg-yellow-500",
  done: "bg-green-500",
};

export function isAgentBusy(status: AgentStatus) {
  return status === "thinking" || status === "working";
}
