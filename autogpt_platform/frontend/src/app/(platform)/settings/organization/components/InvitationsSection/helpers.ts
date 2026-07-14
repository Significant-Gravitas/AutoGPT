// Spell out the teams an invitee is pre-assigned to (e.g. "Marketing, Design")
// rather than a bare count. Any id that can't be resolved to a name (a team the
// viewer can't see, or one deleted since the invite) falls back to a count so
// the row never silently drops an assignment.
export function formatAssignedTeams(
  teamIds: string[],
  teamNameById: Map<string, string>,
): string {
  const names: string[] = [];
  let unresolved = 0;
  for (const id of teamIds) {
    const name = teamNameById.get(id);
    if (name) names.push(name);
    else unresolved += 1;
  }

  if (names.length === 0) {
    return `${teamIds.length} ${teamIds.length === 1 ? "team" : "teams"}`;
  }
  if (unresolved === 0) {
    return names.join(", ");
  }
  return `${names.join(", ")}, +${unresolved} ${
    unresolved === 1 ? "team" : "teams"
  }`;
}
