/**
 * Pages are fetched by offset, so a run started between two page loads can
 * come back on both. Rows are keyed by run id, so each run is listed once.
 */
export function uniqueRunsByID<T extends { id: string }>(runs: T[]) {
  const seen = new Set<string>();
  return runs.filter((run) => {
    if (seen.has(run.id)) return false;
    seen.add(run.id);
    return true;
  });
}
