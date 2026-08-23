export function formatLearnedAt(learnedAt: string | Date): string {
  const date = learnedAt instanceof Date ? learnedAt : new Date(learnedAt);
  if (Number.isNaN(date.getTime())) return "recently";
  return date.toLocaleDateString(undefined, {
    day: "numeric",
    month: "short",
    year: "numeric",
  });
}
