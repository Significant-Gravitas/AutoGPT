import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";

import type { PlanChoiceDetails } from "./helpers";

interface Props {
  plan: PlanChoiceDetails;
  actionLabel: string;
  variant: "primary" | "outline";
  isLoading: boolean;
  isDisabled: boolean;
  onSelect: () => void;
}

export function PlanChoice({
  plan,
  actionLabel,
  variant,
  isLoading,
  isDisabled,
  onSelect,
}: Props) {
  return (
    <li className="flex flex-col gap-3 rounded-[18px] border border-zinc-200 bg-white p-5 shadow-[0_1px_2px_rgba(15,15,20,0.04)]">
      <Text variant="body-medium" as="h3" className="text-textBlack">
        {plan.label}
      </Text>
      <div className="flex flex-col gap-1">
        <p className="flex items-baseline gap-1.5">
          <Text variant="h4" as="span" className="text-textBlack">
            {plan.amount}
          </Text>
          <Text variant="small" as="span" tone="muted">
            / {plan.cadence}
          </Text>
        </p>
        <Text variant="small" tone="secondary">
          {plan.description}
        </Text>
      </div>
      <Button
        variant={variant}
        size="small"
        className="mt-auto w-full"
        onClick={onSelect}
        disabled={isDisabled}
        loading={isLoading}
      >
        {actionLabel}
      </Button>
    </li>
  );
}
