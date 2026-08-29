function TrackingBadge({
  trackingType,
}: {
  trackingType: string | null | undefined;
}) {
  const colors: Record<string, string> = {
    cost_usd: "bg-green-500/10 text-green-700",
    tokens: "bg-blue-500/10 text-blue-700",
    characters: "bg-purple-500/10 text-purple-700",
    sandbox_seconds: "bg-orange-500/10 text-orange-700",
    walltime_seconds: "bg-orange-500/10 text-orange-700",
    items: "bg-pink-500/10 text-pink-700",
    per_run: "bg-muted text-muted-foreground",
  };
  const label = trackingType || "per_run";
  return (
    <span
      className={`inline-block rounded px-1.5 py-0.5 text-[10px] font-medium ${colors[label] || colors.per_run}`}
    >
      {label}
    </span>
  );
}

export { TrackingBadge };
