"use client";

import { Text } from "@/components/atoms/Text/Text";
import { Collapsible } from "@/components/molecules/Collapsible/Collapsible";
import { OnboardingStep } from "@/lib/autogpt-server-api";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { getEarnRows, TaskGroup } from "../helpers";
import {
  CheckmarkBadge01Icon,
  CircleIcon,
  CreditCardIcon,
} from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

interface Props {
  groups: TaskGroup[];
  completedSteps: OnboardingStep[] | undefined;
  formattedCredits: string;
  onAddCredits: () => void;
}

export function WalletCompactPanel({
  groups,
  completedSteps,
  formattedCredits,
  onAddCredits,
}: Props) {
  const isPaymentEnabled = useGetFlag(Flag.ENABLE_PLATFORM_PAYMENT);
  const rows = getEarnRows(groups, completedSteps);

  return (
    <div className="flex flex-col gap-1">
      <div className="flex items-center justify-between gap-3 px-3 py-2">
        <Text variant="body-medium">Automation credits</Text>
        <span className="font-sans text-base font-semibold text-zinc-900">
          {formattedCredits}
        </span>
      </div>

      {isPaymentEnabled && (
        <button
          type="button"
          onClick={onAddCredits}
          className="flex items-center justify-center gap-3 rounded-large bg-zinc-100 px-3 py-2.5 text-center transition-colors hover:bg-zinc-200"
        >
          <Icon icon={CreditCardIcon} size={20} className="text-zinc-700" />
          <Text variant="body-medium">Add credits</Text>
        </button>
      )}

      <Collapsible
        defaultOpen
        triggerClassName="px-3 pb-1.5 pt-2"
        trigger={<Text variant="body-medium">Earn credits</Text>}
      >
        <div className="max-h-[20rem] overflow-y-auto">
          {rows.map((row) => (
            <div
              key={row.key}
              className="flex items-start justify-between gap-3 px-3 py-1.5"
            >
              <div className="flex min-w-0 items-start gap-2.5">
                <span className="mt-0.5 shrink-0">
                  {row.done ? (
                    <Icon
                      icon={CheckmarkBadge01Icon}
                      size={18}
                      className="text-[#00a656]"
                      aria-label="completed"
                    />
                  ) : (
                    <Icon
                      icon={CircleIcon}
                      size={16}
                      className="text-zinc-400"
                      aria-label="pending"
                    />
                  )}
                </span>
                <Text variant="body">{row.label}</Text>
              </div>
              <span className="shrink-0 font-sans text-sm text-zinc-500">
                {row.done ? "Done" : `$${row.amount.toFixed(2)}`}
              </span>
            </div>
          ))}
        </div>
      </Collapsible>
    </div>
  );
}
