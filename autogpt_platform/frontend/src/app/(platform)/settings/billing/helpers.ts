export const EASE_OUT = [0.16, 1, 0.3, 1] as const;

const CURRENCY_FORMATTER = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  minimumFractionDigits: 2,
  maximumFractionDigits: 2,
});

export function formatCents(cents: number): string {
  return CURRENCY_FORMATTER.format(cents / 100);
}

export function formatRelativeReset(target: Date | string | undefined | null): {
  prefix: string;
  value: string;
} {
  if (!target) return { prefix: "Resets", value: "—" };
  const date = target instanceof Date ? target : new Date(target);
  if (Number.isNaN(date.getTime())) return { prefix: "Resets", value: "—" };
  const diff = date.getTime() - Date.now();
  if (diff <= 0) return { prefix: "Resets", value: "soon" };
  const hours = Math.floor(diff / (1000 * 60 * 60));
  if (hours < 24) {
    const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));
    return { prefix: "Resets in", value: `${hours}h ${minutes}m` };
  }
  return {
    prefix: "Resets",
    value: date.toLocaleString(undefined, {
      weekday: "short",
      hour: "numeric",
      minute: "2-digit",
      timeZoneName: "short",
    }),
  };
}

export function formatShortDate(
  value: Date | string | number | undefined | null,
): string {
  if (value === undefined || value === null || value === "") return "—";
  const date = value instanceof Date ? value : new Date(value);
  if (Number.isNaN(date.getTime())) return "—";
  return date.toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

// Tier picker config — three user-visible plans matching the launch doc
// (Pro $50, Max $320, Team contact-sales). BASIC + ENTERPRISE are reserved
// internal slots and never offered as picker cards. Order is low → high so
// the grid reads left-to-right as upgrade direction.
export interface TierInfo {
  key: "PRO" | "MAX" | "BUSINESS";
  label: string;
  description: string;
  contactSales?: boolean;
}

export const PLAN_TIERS: TierInfo[] = [
  {
    key: "PRO",
    label: "Pro",
    description: "Base AutoPilot capacity — solopreneurs, freelancers, power users.",
  },
  {
    key: "MAX",
    label: "Max",
    description: "~8.5x Pro headroom — heavier workflows, agencies, small teams.",
  },
  {
    key: "BUSINESS",
    label: "Team",
    description: "Custom allowances, seats, and support. Talk to sales.",
    contactSales: true,
  },
];

// Same ordering used to compare two tiers for upgrade vs downgrade direction.
// Includes NO_TIER + ENTERPRISE so comparisons against admin-granted users
// resolve correctly even though those tiers aren't picker-rendered.
export const TIER_ORDER = [
  "NO_TIER",
  "BASIC",
  "PRO",
  "MAX",
  "BUSINESS",
  "ENTERPRISE",
] as const;

export function getTierLabel(tierKey: string): string {
  if (tierKey === "NO_TIER") return "No active subscription";
  if (tierKey === "ENTERPRISE") return "Enterprise";
  return PLAN_TIERS.find((t) => t.key === tierKey)?.label ?? tierKey;
}

export function formatTierCost(cents: number, contactSales?: boolean): string {
  if (contactSales) return "Contact us";
  if (cents === 0) return "Free";
  return `$${(cents / 100).toFixed(0)}/mo`;
}

// Render a tier's rate-limit badge relative to the lowest visible tier so the
// UI doesn't have to hard-code backend multiplier defaults. Returns null for
// the lowest tier (it's the baseline) and for tiers absent from the payload.
export function formatRelativeMultiplier(
  tierKey: string,
  tierMultipliers: Record<string, number>,
): string | null {
  const mine = tierMultipliers[tierKey];
  if (mine === undefined || mine <= 0) return null;
  const visible = Object.values(tierMultipliers).filter((v) => v > 0);
  if (visible.length === 0) return null;
  const min = Math.min(...visible);
  const label = (mine / min).toFixed(1);
  if (label === "1.0") return null;
  return `${label}x rate limits`;
}
