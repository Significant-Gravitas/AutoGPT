"use client";

import { getExpertRoleLabel } from "@/services/experts/expert-role-label";

import type { RecommendedExpert } from "@/app/api/__generated__/models/recommendedExpert";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { cn } from "@/lib/utils";
import { Tick02Icon } from "@hugeicons/core-free-icons";

interface Props {
  expert: RecommendedExpert;
  isHired: boolean;
  isHiringThis: boolean;
  isIncluded: boolean;
  disabled: boolean;
  onHire: () => void;
  onToggleIncluded: () => void;
}

export const EXPERT_CARD_CLASS =
  "flex h-full flex-col gap-2 rounded-2xl border bg-white p-4 text-left transition-colors";

export function RecommendedExpertCard({
  expert,
  isHired,
  isHiringThis,
  isIncluded,
  disabled,
  onHire,
  onToggleIncluded,
}: Props) {
  const isLeftOut = !isHired && !isIncluded;
  return (
    <div
      className={cn(
        EXPERT_CARD_CLASS,
        "relative isolate",
        isHired ? "border-zinc-900" : "border-zinc-200",
      )}
      data-testid="hire-step-expert"
      data-hired={isHired || undefined}
    >
      {isHired ? null : (
        // The wizard's own pick badge (SelectableCard), as a toggle: ticked
        // cards are what the step's primary action hires. z-10 lifts it over the
        // faded header; the card's `isolate` keeps it under the list's edge fades.
        <button
          type="button"
          aria-pressed={isIncluded}
          aria-label={`Include ${expert.name}`}
          disabled={disabled}
          onClick={onToggleIncluded}
          className="absolute right-2 top-2 z-10 flex size-7 items-center justify-center rounded-full disabled:cursor-not-allowed"
        >
          <span
            className={cn(
              "flex size-4 items-center justify-center rounded-full border transition-colors",
              isIncluded
                ? "border-zinc-900 bg-zinc-900"
                : "border-zinc-300 bg-white",
            )}
          >
            {isIncluded ? (
              <Icon icon={Tick02Icon} size={10} className="text-white" />
            ) : null}
          </span>
        </button>
      )}
      <div
        className={cn(
          "flex items-center gap-3 pr-6 transition-opacity",
          isLeftOut && "opacity-50",
        )}
      >
        <ExpertAvatar
          name={expert.name}
          avatarUrl={expert.avatar_url ?? null}
          size={40}
        />
        <div className="min-w-0">
          <Text variant="body-medium" tone="primary" className="truncate">
            {expert.name}
          </Text>
          <Text variant="small" tone="muted" className="truncate">
            {getExpertRoleLabel(expert.role)}
          </Text>
        </div>
      </div>
      {expert.reason ? (
        <Text
          variant="body"
          tone="secondary"
          className={cn(
            "line-clamp-2 text-pretty transition-opacity",
            isLeftOut && "opacity-50",
          )}
        >
          {expert.reason}
        </Text>
      ) : null}
      {isHired ? (
        <Text
          variant="small"
          as="span"
          className="mt-auto inline-flex items-center gap-1 self-start pt-1 text-zinc-900"
        >
          <Icon icon={Tick02Icon} size={14} />
          Hired
        </Text>
      ) : (
        <Button
          variant="primary"
          size="xs"
          className="mt-auto self-start rounded-full"
          disabled={disabled}
          loading={isHiringThis}
          onClick={onHire}
        >
          Hire
        </Button>
      )}
    </div>
  );
}
