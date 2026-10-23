// TODO(tokens): no role for these colours yet, so they stay raw:
// bg-green-500 #16A34A, bg-orange-500 #FE8700, bg-pink-500 #EC4899,
// text-blue-700 #1D4ED8, text-orange-700 #B46000, text-pink-700 #A8336D,
// text-purple-700 #5424AE.

function TrackingBadge({
  trackingType,
}: {
  trackingType: string | null | undefined;
}) {
  const colors: Record<string, string> = {
    cost_usd: "bg-green-500/10 text-success-foreground",
    tokens: "bg-info/10 text-blue-700",
    characters: "bg-brand/10 text-purple-700",
    sandbox_seconds: "bg-orange-500/10 text-orange-700",
    walltime_seconds: "bg-orange-500/10 text-orange-700",
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
