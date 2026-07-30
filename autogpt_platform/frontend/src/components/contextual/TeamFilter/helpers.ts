// Sentinel Select values for the two non-team filter choices.
export const TEAM_FILTER_ALL = "__all__";
export const TEAM_FILTER_ORG_HOME = "__org_home__";

// Client-side predicate: does a row with this team id pass the active filter?
// "All" passes everything; "Organization" passes rows with no team (org-home);
// a team id passes only that team's rows.
export function matchesTeamFilter(
  rowTeamId: string | null | undefined,
  filter: string,
): boolean {
  if (filter === TEAM_FILTER_ALL) return true;
  if (filter === TEAM_FILTER_ORG_HOME) return !rowTeamId;
  return rowTeamId === filter;
}
