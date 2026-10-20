"use client";

import {
  ArrowRight02Icon,
  SparklesIcon,
  Tick02Icon,
} from "@hugeicons/core-free-icons";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { formatOfferPrice } from "@/services/usageExperience/helpers";

interface Props {
  tier: "PRO" | "MAX";
  price?: number;
  currency?: string;
  cycle?: "monthly" | "yearly";
  usageMultiplier?: number;
  storageMultiplier?: number;
  freshUsage?: boolean;
  actionLabel?: string;
  onUpgrade: () => void;
  disabled?: boolean;
}

export function PlanOffer({
  tier,
  price,
  currency = "USD",
  cycle = "monthly",
  usageMultiplier,
  storageMultiplier,
  freshUsage = false,
  actionLabel,
  onUpgrade,
  disabled = false,
}: Props) {
  const name = tier === "PRO" ? "Pro" : "Max";
  const amount = formatOfferPrice(price, currency);
  const features =
    tier === "PRO"
      ? [
          freshUsage
            ? "A full daily and weekly allowance once payment goes through"
            : "Experts for content, sales and research, with Otto coordinating",
          freshUsage
            ? "Your chats, workflows and files stay where they are"
            : "Workflows that keep working in the background on a schedule",
        ]
      : [
          usageMultiplier
            ? `${usageMultiplier}× the usage of Pro`
            : "More room for your daily and weekly work",
          storageMultiplier
            ? `${storageMultiplier}× the workspace storage`
            : "More workspace storage",
          "Priority support",
        ];
  return (
    <section
      aria-label={`${name} plan`}
      className="rounded-2xl border border-purple-200 bg-gradient-to-br from-purple-50 via-white to-blue-50 p-5 sm:p-6"
    >
      <div className="mb-5 flex items-start justify-between gap-3">
        <div className="flex min-w-0 items-center gap-3">
          <span className="hidden size-10 shrink-0 items-center justify-center rounded-xl border border-purple-200 bg-white text-purple-600 min-[360px]:flex">
            <Icon icon={SparklesIcon} size={20} />
          </span>
          <div>
            <Text variant="h4" className="!text-xl">
              {name}
            </Text>
            <Text variant="small" className="!text-zinc-500">
              Billed {cycle}
            </Text>
          </div>
        </div>
        {amount && (
          <div className="shrink-0 text-right">
            <Text variant="h4" className="!text-2xl">
              {amount}
            </Text>
            <Text variant="small" className="!text-zinc-500">
              per {cycle === "yearly" ? "year" : "month"} + tax
            </Text>
          </div>
        )}
      </div>
      <ul className="mb-5 space-y-2.5">
        {features.map((feature) => (
          <li
            key={feature}
            className="flex gap-2.5 text-sm leading-5 text-zinc-700"
          >
            <Icon
              icon={Tick02Icon}
              size={16}
              className="mt-0.5 shrink-0 text-purple-600"
            />
            {feature}
          </li>
        ))}
      </ul>
      <Button
        className="w-full"
        onClick={onUpgrade}
        disabled={disabled}
        rightIcon={<Icon icon={ArrowRight02Icon} size={16} />}
      >
        {actionLabel ??
          (tier === "MAX"
            ? "Upgrade to Max"
            : freshUsage
              ? `Start Pro now${amount ? ` · ${amount}` : ""}`
              : "Start Pro")}
      </Button>
      <Text
        variant="small"
        className="mt-3 text-center !text-xs !text-zinc-500"
      >
        {freshUsage
          ? "You’ll see the full charge before you pay."
          : tier === "MAX"
            ? "What you’ve already used counts toward Max’s higher limit."
            : "Review your plan and payment before confirming."}
      </Text>
    </section>
  );
}
