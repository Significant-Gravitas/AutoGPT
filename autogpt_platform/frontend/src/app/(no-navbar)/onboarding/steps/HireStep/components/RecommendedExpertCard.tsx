"use client";

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
  disabled: boolean;
  onHire: () => void;
}

export const EXPERT_CARD_CLASS =
  "flex h-full flex-col gap-2 rounded-2xl border bg-white p-4 text-left transition-colors";

export function RecommendedExpertCard({
  expert,
  isHired,
  isHiringThis,
  disabled,
  onHire,
}: Props) {
  return (
    <div
      className={cn(
        EXPERT_CARD_CLASS,
        isHired ? "border-zinc-900" : "border-zinc-200",
      )}
      data-testid="hire-step-expert"
      data-hired={isHired || undefined}
    >
      <div className="flex items-center gap-3">
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
            {expert.role}
          </Text>
        </div>
      </div>
      {expert.reason ? (
        <Text
          variant="body"
          tone="secondary"
          className="line-clamp-2 text-pretty"
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
