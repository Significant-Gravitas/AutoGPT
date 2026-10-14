const STATUS_DOT_CLASS: Record<string, string> = {
  backlog: "bg-zinc-400",
  in_progress: "bg-sky-500",
  done: "bg-emerald-500",
};

export function statusDotClass(columnKey: string | null): string {
  if (!columnKey) return "bg-zinc-300";
  return STATUS_DOT_CLASS[columnKey] ?? "bg-violet-400";
}
