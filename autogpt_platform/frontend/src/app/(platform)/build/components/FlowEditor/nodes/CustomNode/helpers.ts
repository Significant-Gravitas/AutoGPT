import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";

export const nodeStyleBasedOnStatus: Record<AgentExecutionStatus, string> = {
  INCOMPLETE: "ring-slate-300 bg-slate-300",
  QUEUED: " ring-blue-300 bg-blue-300",
  RUNNING: "ring-amber-300 bg-amber-300",
  REVIEW: "ring-orange-300 bg-orange-300",
  COMPLETED: "ring-green-300 bg-green-300",
  TERMINATED: "ring-orange-300 bg-orange-300 ",
  FAILED: "ring-red-300 bg-red-300",
};
