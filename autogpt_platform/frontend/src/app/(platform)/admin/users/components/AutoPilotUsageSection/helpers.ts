import { getPlanName } from "../../helpers";

// TRIAL is set by billing, never assigned by hand.
const ASSIGNABLE_TIERS = [
  "NO_TIER",
  "BASIC",
  "PRO",
  "MAX",
  "BUSINESS",
  "ENTERPRISE",
] as const;

export type ResetScope = "daily" | "both";

export const RESET_SCOPE_OPTIONS = [
  { value: "daily", label: "Daily usage" },
  { value: "both", label: "Daily and weekly usage" },
];

export function isAssignableTier(tier: string) {
  return (ASSIGNABLE_TIERS as readonly string[]).includes(tier);
}

export function getPlanOptions(multipliers?: Record<string, number>) {
  return ASSIGNABLE_TIERS.map((tier) => ({
    value: tier,
    label: getPlanOptionLabel(tier, multipliers),
  }));
}

// Multipliers come live from the admin endpoint; a tier missing from the map
// gets a generic label rather than a made-up number.
function getPlanOptionLabel(
  tier: string,
  multipliers?: Record<string, number>,
) {
  if (tier === "NO_TIER") return "No plan — no AutoPilot access";
  const multiplier = multipliers?.[tier];
  if (multiplier === undefined || multiplier <= 0) {
    return `${getPlanName(tier)} — tier limits`;
  }
  return `${getPlanName(tier)} — ${Math.round(multiplier * 100) / 100}× base limits`;
}
