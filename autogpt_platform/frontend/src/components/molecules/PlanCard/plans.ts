export const PRO_USD_MONTHLY = 50;
export const MAX_USD_MONTHLY = 320;
// Fraction of the full monthly price paid when billed yearly. 0.85 = 15% off.
export const YEARLY_PRICE_FACTOR = 0.85;
export const TEAM_INTAKE_FORM_URL = "https://tally.so/r/2Eb9zj";

export const PLAN_KEYS = {
  PRO: "PRO",
  MAX: "MAX",
  TEAM: "TEAM",
  BUSINESS: "BUSINESS",
} as const;
export type PlanKey = (typeof PLAN_KEYS)[keyof typeof PLAN_KEYS];

export interface PlanDef {
  key: PlanKey;
  name: string;
  usage: string | null;
  usdMonthly: number | null;
  // Explicit yearly price in USD; when provided, computePlanPricing uses it
  // instead of usdMonthly * YEARLY_PRICE_FACTOR * 12. Lets API-driven surfaces
  // (PaywallGate) display the actual Stripe yearly amount.
  usdYearly?: number | null;
  description: string;
  features: string[];
  cta: string;
  highlighted: boolean;
  badge: string | null;
  buttonVariant: "primary" | "secondary";
}

export const PLANS: PlanDef[] = [
  {
    key: PLAN_KEYS.PRO,
    name: "Pro",
    usage: "1x",
    usdMonthly: PRO_USD_MONTHLY,
    description:
      "For individuals getting started with dependable everyday automation.",
    features: [
      "Access to virtually any leading AI model",
      "Agents work non-stop in the background",
      "Visual agent builder and chat-based agent creation",
      "File-aware agents that can work with your documents",
      "End-to-end agent management and run visibility",
      "Scheduled and event-based triggers",
    ],
    cta: "Get Pro",
    highlighted: false,
    badge: null,
    buttonVariant: "secondary",
  },
  {
    key: PLAN_KEYS.MAX,
    name: "Max",
    usage: "8.5x",
    usdMonthly: MAX_USD_MONTHLY,
    description: "For users who are serious about getting more work done.",
    features: [
      "Includes everything in Pro",
      "8.5x Pro usage",
      "5x file storage",
      "Early access to features before anyone else",
      "Expand our extensive integration library",
      "Priority support",
      "Help drive the roadmap for new features",
    ],
    cta: "Upgrade to Max",
    highlighted: true,
    badge: "Best value",
    buttonVariant: "primary",
  },
  {
    key: PLAN_KEYS.TEAM,
    name: "Team",
    usage: null,
    usdMonthly: null,
    description:
      "For teams that need collaboration, controls, and custom usage.",
    features: [
      "Multi-user workspaces",
      "Admin controls & roles",
      "Centralized billing & seat management",
      "Collaboration & approvals",
      "Security & compliance options",
      "Secure access for shared integrations",
    ],
    cta: "Contact sales",
    highlighted: false,
    badge: "Coming soon",
    buttonVariant: "secondary",
  },
];

// Backend-tier → static visual metadata. Used by API-driven surfaces
// (PaywallGate) which derive prices from /credits/subscription and need only
// the descriptive content (features/usage/highlight/cta).
export type BackendTierKey = "PRO" | "MAX" | "BUSINESS";

export const PLAN_METADATA: Record<
  BackendTierKey,
  Omit<PlanDef, "usdMonthly" | "usdYearly">
> = {
  PRO: {
    key: PLAN_KEYS.PRO,
    name: "Pro",
    usage: "1x",
    description:
      "For individuals getting started with dependable everyday automation.",
    features: [
      "Access to virtually any leading AI model",
      "Agents work non-stop in the background",
      "Visual agent builder and chat-based agent creation",
      "File-aware agents that can work with your documents",
      "End-to-end agent management and run visibility",
      "Scheduled and event-based triggers",
    ],
    cta: "Upgrade to Pro",
    highlighted: false,
    badge: null,
    buttonVariant: "secondary",
  },
  MAX: {
    key: PLAN_KEYS.MAX,
    name: "Max",
    usage: "8.5x",
    description: "For users who are serious about getting more work done.",
    features: [
      "Includes everything in Pro",
      "8.5x Pro usage",
      "5x file storage",
      "Early access to features before anyone else",
      "Expand our extensive integration library",
      "Priority support",
      "Help drive the roadmap for new features",
    ],
    cta: "Upgrade to Max",
    highlighted: true,
    badge: "Best value",
    buttonVariant: "primary",
  },
  BUSINESS: {
    key: PLAN_KEYS.BUSINESS,
    name: "Team",
    usage: null,
    description:
      "For teams and heavy workloads that need expanded capacity and controls.",
    features: [
      "Includes everything in Max",
      "Highest AutoPilot capacity",
      "Priority support and onboarding",
      "Advanced security & compliance options",
      "Dedicated account contact",
    ],
    // Team is contact-sales — the click handler diverts to TEAM_INTAKE_FORM_URL
    // rather than POSTing to /credits/subscription, so the CTA reflects that.
    cta: "Talk to sales",
    highlighted: false,
    badge: null,
    buttonVariant: "secondary",
  },
};
