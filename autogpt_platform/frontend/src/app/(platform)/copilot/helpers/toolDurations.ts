// How long each tool call took, by tool call id. A saved chat brings the
// time the backend measured; a live one notes how long this tab saw the step
// run, until the saved time replaces it.
const MAX_REMEMBERED = 2_000;
const durations = new Map<string, number>();

export function noteToolDuration(toolCallID: string, milliseconds: number) {
  if (!toolCallID || !Number.isFinite(milliseconds) || milliseconds < 0) return;
  durations.delete(toolCallID);
  durations.set(toolCallID, milliseconds);
  if (durations.size > MAX_REMEMBERED) {
    const oldest = durations.keys().next().value;
    if (oldest !== undefined) durations.delete(oldest);
  }
}

export function getToolDuration(toolCallID: string) {
  return durations.get(toolCallID) ?? null;
}

/** The steps' time added up; null when none of them was timed. */
export function totalToolDuration(toolCallIDs: string[]) {
  let total = 0;
  let timed = 0;
  for (const id of toolCallIDs) {
    const duration = durations.get(id);
    if (duration === undefined) continue;
    total += duration;
    timed += 1;
  }
  return timed > 0 ? total : null;
}

/** "0.4s", "4.2s", "38s", "2m 5s". */
export function formatToolDuration(milliseconds: number) {
  const seconds = milliseconds / 1000;
  if (seconds < 10) return `${Math.max(0.1, seconds).toFixed(1)}s`;
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const minutes = Math.floor(seconds / 60);
  const rest = Math.round(seconds % 60);
  return rest ? `${minutes}m ${rest}s` : `${minutes}m`;
}
