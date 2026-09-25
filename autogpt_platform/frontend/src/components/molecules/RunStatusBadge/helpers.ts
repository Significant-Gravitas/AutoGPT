const GUIDANCE: Record<string, string> = {
  COMPLETED: "Execution finished. Open the output to review the result.",
  FAILED:
    "Execution failed. Open run details to inspect the cause before retrying.",
  RUNNING: "Execution is in progress. Open run details to follow progress.",
  QUEUED: "Execution is waiting to start. Open run details to check progress.",
  REVIEW:
    "Execution is waiting for a decision. Review the request before approving or rejecting it.",
  TERMINATED: "Execution stopped. Open run details to inspect where it ended.",
  INCOMPLETE:
    "Execution has not completed. Open run details to inspect what remains.",
};

export function getRunStatusGuidance(status: string): string {
  return (
    GUIDANCE[status.toUpperCase()] ??
    "No status details available. Open run details for more information."
  );
}
