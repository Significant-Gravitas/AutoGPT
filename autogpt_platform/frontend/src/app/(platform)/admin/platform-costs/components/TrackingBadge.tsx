function trackingBadge(trackingType: string | null | undefined) {
  const colors: Record<string, string> = {
    cost_usd:
      "bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400",
    tokens: "bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-400",
    duration_seconds:
      "bg-orange-100 text-orange-800 dark:bg-orange-900/30 dark:text-orange-400",
    characters:
      "bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-400",
    sandbox_seconds:
      "bg-orange-100 text-orange-800 dark:bg-orange-900/30 dark:text-orange-400",
    walltime_seconds:
      "bg-orange-100 text-orange-800 dark:bg-orange-900/30 dark:text-orange-400",
    per_run: "bg-gray-100 text-gray-800 dark:bg-gray-900/30 dark:text-gray-400",
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
