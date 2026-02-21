/**
 * Tools that take a long time to execute (several minutes).
 * These tools will automatically show the mini-game while executing.
 *
 * This list matches the backend tools where is_long_running=True.
 */
export const LONG_RUNNING_TOOLS = [
  "create_agent",
  "edit_agent",
  "customize_agent",
] as const;

export type LongRunningToolName = (typeof LONG_RUNNING_TOOLS)[number];

/**
 * Check if a tool is marked as long-running.
 */
export function isLongRunningTool(toolName: string): boolean {
  return LONG_RUNNING_TOOLS.includes(toolName as LongRunningToolName);
}
