function TrackingBadge({
  trackingType,
}: {
  trackingType: string | null | undefined;
}) {
  const colors: Record<string, string> = {
    cost_usd: "bg-success/10 text-success-foreground",
    tokens: "bg-info/10 text-info-foreground",
    characters: "bg-brand/10 text-accent-strong",
    // oxlint-disable-next-line design-system/no-raw-colors -- orange has no role; amber would change the hue (deferred)
    sandbox_seconds: "bg-orange-500/10 text-orange-700",
    // oxlint-disable-next-line design-system/no-raw-colors -- orange has no role; amber would change the hue (deferred)
    walltime_seconds: "bg-orange-500/10 text-orange-700",
    // oxlint-disable-next-line design-system/no-raw-colors -- pink has no role; a role would change the hue (deferred)
    items: "bg-pink-500/10 text-pink-700",
    per_run: "bg-muted text-muted-foreground",
  };
  const label = trackingType || "per_run";
  return (
    <span
      className={`inline-block rounded px-1.5 py-0.5 text-tiny font-medium leading-inherit ${colors[label] || colors.per_run}`}
    >
      {label}
    </span>
  );
}

export { TrackingBadge };
