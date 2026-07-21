"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { ArrowRightIcon, LightbulbIcon } from "@phosphor-icons/react";

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
    <div className="rounded-xl border border-amber-200 bg-amber-50/50 p-4">
      <div className="flex items-start gap-3">
        <LightbulbIcon
          size={20}
          weight="fill"
          className="mt-0.5 text-amber-600"
        />
        <div className="flex-1 space-y-3">
          <div>
            <Text variant="body-medium" className="font-medium text-slate-900">
              {goalType === "unachievable"
                ? "Goal cannot be accomplished"
                : "Goal needs more detail"}
            </Text>
            <Text variant="small" className="text-slate-600">
              {reason || message}
            </Text>
          </div>

          <div className="rounded-lg border border-amber-300 bg-white p-3">
            <Text variant="small" className="mb-1 font-semibold text-amber-800">
              Suggested alternative:
            </Text>
            <Text variant="body-medium" className="text-slate-900">
              {suggestedGoal}
            </Text>
          </div>

          <Button
            onClick={() => onUseSuggestedGoal(suggestedGoal)}
            variant="primary"
          >
            <span className="inline-flex items-center gap-1.5">
              Use this goal <ArrowRightIcon size={14} weight="bold" />
            </span>
          </Button>
        </div>
      </div>
    </div>
  );
}
