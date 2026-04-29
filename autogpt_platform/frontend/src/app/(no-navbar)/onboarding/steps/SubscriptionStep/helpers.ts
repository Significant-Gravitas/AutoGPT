export const PRO_USD_MONTHLY = 50;
export const MAX_USD_MONTHLY = 320;
// Fraction of the full monthly price paid when billed yearly. 0.85 = 15% off.
export const YEARLY_PRICE_FACTOR = 0.85;
export const TEAM_INTAKE_FORM_URL = "https://tally.so/r/2Eb9zj";

export const PLAN_KEYS = {
  PRO: "PRO",
  MAX: "MAX",
  TEAM: "TEAM",
} as const;
export type PlanKey = (typeof PLAN_KEYS)[keyof typeof PLAN_KEYS];

export interface PlanDef {
  key: PlanKey;
  name: string;
  usage: string | null;
  usdMonthly: number | null;
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
      "1x usage allowance",
      "Up to 1,000 agent runs / month",
      "Access to core models",
      "Core agent library",
      "Standard support",
    ],
    cta: "Get Pro",
    highlighted: false,
    badge: null,
    buttonVariant: "secondary",
  },
  {
    key: PLAN_KEYS.MAX,
    name: "Max",
    usage: "7x",
    usdMonthly: MAX_USD_MONTHLY,
    description:
      "For power users running more workflows and higher-volume automations.",
    features: [
      "7x usage allowance",
      "Up to 10,000 agent runs / month",
      "Access to premium models",
      "Advanced workflows & tools",
      "Priority processing",
      "Faster support",
      "Custom integrations (beta)",
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
      "Shared billing & usage pools",
      "Collaboration & approvals",
      "Security & compliance options",
      "Premium support",
    ],
    cta: "Contact sales",
    highlighted: false,
    badge: "Coming soon",
    buttonVariant: "secondary",
  },
];
