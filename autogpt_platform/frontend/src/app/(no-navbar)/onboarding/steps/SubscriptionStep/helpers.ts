export const PRO_USD_MONTHLY = 50;
export const MAX_USD_MONTHLY = 320;
export const YEARLY_DISCOUNT = 0.85;
export const TEAM_INTAKE_FORM_URL = "https://tally.so/r/2Eb9zj";

export interface PlanDef {
  key: string;
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
    key: "PRO",
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
    key: "BUSINESS",
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
    key: "ENTERPRISE",
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
