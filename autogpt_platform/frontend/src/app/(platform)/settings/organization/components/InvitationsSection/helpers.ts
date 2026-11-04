import { ApiError } from "@/lib/autogpt-server-api/helpers";

// The backend answers 409 when the organization has no free seat (or no team
// plan at all). Its message says which and what to do, so it is shown as is.
export function getSeatRefusal(error: unknown): string | null {
  if (!(error instanceof ApiError) || error.status !== 409) return null;
  return error.message || null;
}

// Pill labels for the teams an invitee is pre-assigned to. Each resolved team
// name becomes its own label; ids that can't be resolved (a team the viewer
// can't see, or one deleted since the invite) collapse into a single "+N"
// label so the row never silently drops an assignment.
export function assignedTeamLabels(
  teamIds: string[],
  teamNameById: Map<string, string>,
): string[] {
  const labels: string[] = [];
  let unresolved = 0;
  for (const id of teamIds) {
    const name = teamNameById.get(id);
    if (name) labels.push(name);
    else unresolved += 1;
  }

  if (labels.length === 0 && unresolved > 0) {
    return [`${unresolved} ${unresolved === 1 ? "team" : "teams"}`];
  }
  if (unresolved > 0) {
    labels.push(`+${unresolved} ${unresolved === 1 ? "team" : "teams"}`);
  }
  return labels;
}
