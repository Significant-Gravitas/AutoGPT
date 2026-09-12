function ExecutionPathBadge({
  executionPath,
}: {
  executionPath: string | null | undefined;
}) {
  if (!executionPath) return <span className="text-muted-foreground">—</span>;
  const colors: Record<string, string> = {
    anthropic_batch: "bg-green-500/15 text-green-700",
    openai_batch: "bg-green-500/15 text-green-700",
    flex: "bg-blue-500/15 text-blue-700",
    sync_baseline: "bg-muted text-muted-foreground",
    sync: "bg-muted text-muted-foreground",
  };
  // Shorten the long ``anthropic_batch`` / ``openai_batch`` /
  // ``sync_baseline`` labels so the column doesn't dominate the
  // table. The full path is in the title attribute for hover.
  const display =
    executionPath === "anthropic_batch" || executionPath === "openai_batch"
      ? "batch"
      : executionPath === "sync_baseline"
        ? "sync"
        : executionPath;
  return (
    <span
      title={executionPath}
      className={`inline-block rounded px-1.5 py-0.5 text-[10px] font-medium ${colors[executionPath] || colors.sync}`}
    >
      {display}
    </span>
  );
}

export { ExecutionPathBadge };
