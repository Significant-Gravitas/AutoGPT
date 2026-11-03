export function parseDueDate(due: string) {
  const [y, m, d] = due.slice(0, 10).split("-").map(Number);
  return new Date(y, (m ?? 1) - 1, d ?? 1);
}

export function isOverdue(due: string) {
  const today = new Date();
  today.setHours(0, 0, 0, 0);
  return parseDueDate(due) < today;
}

export function formatDue(due: string) {
  return parseDueDate(due).toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
  });
}
