function TrackingBadge({
  trackingType,
}: {
  trackingType: string | null | undefined;
}) {
  const colors: Record<string, string> = {
    cost_usd: "bg-success/10 text-success-foreground",
    tokens: "bg-info/10 text-info-foreground",
    characters: "bg-brand/10 text-accent-strong",
    sandbox_seconds: "bg-orange/10 text-warning-foreground",
    walltime_seconds: "bg-orange/10 text-warning-foreground",
    items: "bg-pink/10 text-pink-foreground",
    per_run: "bg-surface-hover text-foreground-tertiary",
  };
  const label = trackingType || "per_run";
  return (
    <span
      className={`inline-block rounded px-1.5 py-0.5 text-tiny leading-inherit font-medium ${colors[label] || colors.per_run}`}
    >
      {label}
    </span>
  );
}

export { TrackingBadge };
