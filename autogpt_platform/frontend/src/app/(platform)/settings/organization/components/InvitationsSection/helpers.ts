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
