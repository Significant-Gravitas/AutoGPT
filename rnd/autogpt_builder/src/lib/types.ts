import { NodeExecutionResult } from "@/lib/autogpt-server-api";

export type FlowRun = {
  id: string;
  graphID: string;
  graphVersion: number;
  status: "running" | "waiting" | "success" | "failed";
  startTime: number; // unix timestamp (ms)
  endTime: number; // unix timestamp (ms)
  duration: number; // seconds
  totalRunTime: number; // seconds
  nodeExecutionResults: NodeExecutionResult[];
};
