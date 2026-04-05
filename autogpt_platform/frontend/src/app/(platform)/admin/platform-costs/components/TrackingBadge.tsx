function trackingBadge(trackingType: string | null | undefined) {
  const colors: Record<string, string> = {
    cost_usd: "bg-green-100 text-green-800",
    tokens: "bg-blue-100 text-blue-800",
    duration_seconds: "bg-orange-100 text-orange-800",
    characters: "bg-purple-100 text-purple-800",
    sandbox_seconds: "bg-orange-100 text-orange-800",
    walltime_seconds: "bg-orange-100 text-orange-800",
    items: "bg-pink-100 text-pink-800",
    per_run: "bg-gray-100 text-gray-800",
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

export { trackingBadge };
