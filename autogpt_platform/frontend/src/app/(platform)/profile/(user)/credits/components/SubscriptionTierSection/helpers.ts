export interface TierInfo {
  key: string;
  label: string;
  multiplier: string;
  description: string;
}

export const TIERS: TierInfo[] = [
  {
    key: "BASIC",
    label: "Basic",
    multiplier: "1x",
    description: "Base AutoPilot capacity with standard rate limits",
  },
  {
    key: "PRO",
    label: "Pro",
    multiplier: "5x",
    description: "5x AutoPilot capacity — run 5× more tasks per day/week",
  },
  {
    key: "MAX",
    label: "Max",
    multiplier: "20x",
    description: "20x AutoPilot capacity — ideal for power users",
  },
  {
    key: "BUSINESS",
    label: "Business",
    multiplier: "60x",
    description: "60x AutoPilot capacity — ideal for teams and heavy workloads",
  },
];

export const TIER_ORDER = ["BASIC", "PRO", "MAX", "BUSINESS", "ENTERPRISE"];

export function formatCost(cents: number, tierKey: string): string {
  if (cents === 0)
    return tierKey === "BASIC" ? "Free" : "Pricing available soon";
  return `$${(cents / 100).toFixed(2)}/mo`;
}

export function getTierLabel(tierKey: string): string {
  return (
    TIERS.find((t) => t.key === tierKey)?.label ??
    tierKey.charAt(0) + tierKey.slice(1).toLowerCase()
  );
}

export function formatPendingDate(value: Date | string): string {
  const date = value instanceof Date ? value : new Date(value);
  // Pin to en-US so SSR and CSR produce the same string — passing `undefined`
  // picks up the server's locale during prerender and the browser's locale on
  // hydration, which triggers a React hydration mismatch warning.
  return date.toLocaleDateString("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}
