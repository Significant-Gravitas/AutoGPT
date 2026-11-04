// Tasks v1 is behind TASKS_V1. Until the backend serves the flag through
// /api/feature-flags, the build-time override turns it on for local dev.
export function isTasksV1Enabled() {
  return process.env.NEXT_PUBLIC_FORCE_FLAG_TASKS_V1 !== "false";
}
