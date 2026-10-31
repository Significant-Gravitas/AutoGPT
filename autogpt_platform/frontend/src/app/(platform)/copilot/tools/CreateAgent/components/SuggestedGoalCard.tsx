"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { ArrowRight02Icon, BulbIcon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

interface Props {
  message: string;
  suggestedGoal: string;
  reason?: string;
  goalType: string;
  onUseSuggestedGoal: (goal: string) => void;
}

export function SuggestedGoalCard({
  message,
  suggestedGoal,
  reason,
  goalType,
  onUseSuggestedGoal,
}: Props) {
  return (
    <div className="rounded-xl border border-warning-border bg-warning-surface/50 p-4">
      <div className="flex items-start gap-3">
        <Icon
          icon={BulbIcon}
          size={20}
          className="mt-0.5 text-warning-foreground"
        />
        <div className="flex-1 space-y-3">
          <div>
            <Text tone="primary" variant="body-medium" className="font-medium">
              {goalType === "unachievable"
                ? "Goal cannot be accomplished"
                : "Goal needs more detail"}
            </Text>
            <Text tone="secondary" variant="small">
              {reason || message}
            </Text>
          </div>

          <div className="rounded-lg border border-warning-border bg-surface p-3">
            <Text tone="warning" variant="small" className="mb-1 font-semibold">
              Suggested alternative:
            </Text>
            <Text tone="primary" variant="body-medium">
              {suggestedGoal}
            </Text>
          </div>

          <Button
            onClick={() => onUseSuggestedGoal(suggestedGoal)}
            variant="primary"
          >
            <span className="inline-flex items-center gap-1.5">
              Use this goal <Icon icon={ArrowRight02Icon} size={14} />
            </span>
          </Button>
        </div>
      </div>
    </div>
  );
}
