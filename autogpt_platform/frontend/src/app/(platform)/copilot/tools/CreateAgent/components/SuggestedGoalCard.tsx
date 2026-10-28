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
        {/* TODO(tokens): no exact role for amber-600 #D97706 */}
        <Icon icon={BulbIcon} size={20} className="mt-0.5 text-amber-600" />
        <div className="flex-1 space-y-3">
          <div>
            {/* TODO(tokens): no exact role for slate-900 #2A313A */}
            <Text variant="body-medium" className="font-medium text-slate-900">
              {goalType === "unachievable"
                ? "Goal cannot be accomplished"
                : "Goal needs more detail"}
            </Text>
            {/* TODO(tokens): no exact role for slate-600 #5B6A7E */}
            <Text variant="small" className="text-slate-600">
              {reason || message}
            </Text>
          </div>

          {/* TODO(tokens): no exact role for amber-300 #FCD34D */}
          <div className="rounded-lg border border-amber-300 bg-surface p-3">
            {/* TODO(tokens): no exact role for amber-800 #92400E */}
            <Text variant="small" className="mb-1 font-semibold text-amber-800">
              Suggested alternative:
            </Text>
            {/* TODO(tokens): no exact role for slate-900 #2A313A */}
            <Text variant="body-medium" className="text-slate-900">
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
