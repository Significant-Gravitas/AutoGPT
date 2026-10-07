"use client";

import { SparklesIcon, Tick02Icon } from "@hugeicons/core-free-icons";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import type { PlanDef } from "@/components/molecules/PlanCard/plans";

interface Props {
  plans: PlanDef[];
  cycle: "monthly" | "yearly";
  onCycle: (cycle: "monthly" | "yearly") => void;
  onSelect: (tier: string) => void;
  pending: boolean;
  selectedTier: string | null;
}

export function PaywallPlanChoices({
  plans,
  cycle,
  onCycle,
  onSelect,
  pending,
  selectedTier,
}: Props) {
  const team = plans.find((plan) => plan.key === "BUSINESS");
  return (
    <section aria-label="Choose your AutoGPT plan" className="w-full space-y-5">
      <div className="flex justify-center">
        <div
          role="group"
          aria-label="Billing period"
          className="inline-flex rounded-full border border-zinc-200 bg-zinc-100 p-1"
        >
          {(["monthly", "yearly"] as const).map((value) => (
            <Button
              key={value}
              variant="ghost"
              size="small"
              aria-pressed={cycle === value}
              onClick={() => onCycle(value)}
              disabled={pending}
              className={
                cycle === value ? "bg-white shadow-sm" : "text-zinc-500"
              }
            >
              {value === "monthly" ? "Monthly billing" : "Yearly billing"}
            </Button>
          ))}
        </div>
      </div>
      <div className="grid gap-4 sm:grid-cols-2">
        {plans
          .filter((plan) => plan.key !== "BUSINESS")
          .map((plan) => (
            <PlanChoice
              key={plan.key}
              plan={plan}
              cycle={cycle}
              onSelect={onSelect}
              pending={pending}
              selectedTier={selectedTier}
            />
          ))}
      </div>
      {team && (
        <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-zinc-200 px-4 py-3">
          <div>
            <Text variant="small-medium">Team</Text>
            <Text variant="small" tone="secondary">
              Custom capacity for your organization.
            </Text>
          </div>
          <Button
            variant="secondary"
            size="small"
            onClick={() => onSelect(team.key)}
          >
            Talk to sales
          </Button>
        </div>
      )}
      <Text variant="small" tone="secondary" className="text-center text-xs">
        Your conversations and agents stay saved. Cancel anytime.
      </Text>
    </section>
  );
}

function PlanChoice({
  plan,
  cycle,
  onSelect,
  pending,
  selectedTier,
}: { plan: PlanDef } & Omit<Props, "plans" | "onCycle">) {
  const amount = cycle === "yearly" ? plan.usdYearly : plan.usdMonthly;
  const formatted =
    amount == null
      ? null
      : new Intl.NumberFormat("en-US", {
          style: "currency",
          currency: "USD",
          maximumFractionDigits: amount % 1 ? 2 : 0,
        }).format(amount);
  const isPro = plan.key === "PRO";
  const features = isPro
    ? [
        "Daily and weekly usage allowances",
        "Keep building with your saved work",
      ]
    : [
        "More room for demanding work",
        "More file storage and priority support",
      ];
  return (
    <section
      aria-label={`${plan.name} plan`}
      className={`flex flex-col rounded-2xl border p-5 sm:p-6 ${isPro ? "border-purple-200 bg-gradient-to-br from-purple-50 via-white to-blue-50" : "border-zinc-200 bg-white"}`}
    >
      <div className="mb-4 flex items-center gap-2">
        <Icon
          icon={SparklesIcon}
          size={18}
          className={isPro ? "text-purple-600" : "text-zinc-600"}
        />
        <Text variant="large-semibold" as="h2">
          {plan.name}
        </Text>
      </div>
      <Text variant="small" tone="secondary" className="mb-4">
        {isPro
          ? "For the work you do every day."
          : "For bigger ideas and heavier work."}
      </Text>
      <div className="mb-5">
        <Text variant="h3" as="span" className="tracking-tight">
          {formatted ?? "Unavailable"}
        </Text>
        {formatted && (
          <Text variant="small" as="span" tone="secondary">
            {" "}
            / {cycle === "yearly" ? "year" : "month"}
          </Text>
        )}
      </div>
      <ul className="mb-6 space-y-2.5">
        {features.map((feature) => (
          <li className="flex gap-2" key={feature}>
            <Icon
              icon={Tick02Icon}
              size={15}
              className="mt-0.5 shrink-0 text-purple-600"
            />
            <Text variant="small" className="text-zinc-700">
              {feature}
            </Text>
          </li>
        ))}
      </ul>
      <Button
        className="mt-auto w-full"
        variant={isPro ? "primary" : "secondary"}
        onClick={() => onSelect(plan.key)}
        loading={pending && selectedTier === plan.key}
        disabled={pending || !formatted}
      >
        Upgrade to {plan.name}
      </Button>
    </section>
  );
}
