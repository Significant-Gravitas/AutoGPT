import type { DelegationSettings } from "@/app/api/__generated__/models/delegationSettings";
import type { DelegationSettingsUpdate } from "@/app/api/__generated__/models/delegationSettingsUpdate";
import type { SelectOption } from "@/components/atoms/Select/Select";

export const DEFAULT_DELEGATION_SETTINGS: DelegationSettingsUpdate = {
  mode: "auto",
  per_delegation_cap_usd: 2,
  daily_budget_usd: 10,
  ask_before_external: true,
  ask_before_over_cap: true,
  new_experts_ask_first: false,
};

export function withDefaults(
  settings: DelegationSettings | null | undefined,
): DelegationSettingsUpdate {
  return { ...DEFAULT_DELEGATION_SETTINGS, ...settings };
}

export const MODE_OPTIONS: SelectOption[] = [
  { value: "ask_first", label: "Ask first" },
  { value: "auto", label: "Auto" },
  { value: "unsupervised", label: "Unsupervised" },
];

const CAP_AMOUNTS = [0.5, 1, 2, 5, 10];
const BUDGET_AMOUNTS = [5, 10, 25, 50];

function toMoneyOptions(amounts: number[], current: number): SelectOption[] {
  const values = amounts.includes(current)
    ? amounts
    : [...amounts, current].sort((a, b) => a - b);
  return values.map((amount) => ({
    value: String(amount),
    label: `$${amount.toFixed(2)}`,
  }));
}

export function getCapOptions(current: number) {
  return toMoneyOptions(CAP_AMOUNTS, current);
}

export function getBudgetOptions(current: number) {
  return toMoneyOptions(BUDGET_AMOUNTS, current);
}

export type ToggleKey =
  | "ask_before_external"
  | "ask_before_over_cap"
  | "new_experts_ask_first";

export const TOGGLES: { key: ToggleKey; title: string; description: string }[] =
  [
    {
      key: "ask_before_external",
      title: "Ask before sending anything outside the workspace",
      description:
        "Email, Slack, posts and partner notes always wait for you, whatever the mode.",
    },
    {
      key: "ask_before_over_cap",
      title: "Ask before going over the cap",
      description:
        "Otto asks instead of stopping when a hand-off needs more than the per-delegation cap.",
    },
    {
      key: "new_experts_ask_first",
      title: "New experts start in ask-first",
      description:
        "Experts hired this week get approval on every hand-off until you switch them.",
    },
  ];
