"use client";

import type { GetV1GetExecutionDetails200 } from "@/app/api/__generated__/models/getV1GetExecutionDetails200";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { AlertCircleIcon } from "@hugeicons/core-free-icons";

interface Props {
  run: GetV1GetExecutionDetails200;
}

export function RunSummary({ run }: Props) {
  if (!run.stats?.activity_status) return null;

  const correctnessScore = run.stats.correctness_score;

  return (
    <div className="space-y-4">
      <Text variant="body" tone="secondary" unmask={false}>
        {run.stats.activity_status}
      </Text>

      {typeof correctnessScore === "number" && (
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <Text variant="body-medium" as="span" tone="secondary">
              Success Estimate:
            </Text>
            <div className="flex items-center gap-2">
              <div className="relative h-2 w-16 overflow-hidden rounded-full bg-zinc-200">
                <div
                  className={`h-full transition-all ${
                    correctnessScore >= 0.8
                      ? "bg-green-500"
                      : correctnessScore >= 0.6
                        ? "bg-yellow-500"
                        : correctnessScore >= 0.4
                          ? "bg-orange-500"
                          : "bg-red-500"
                  }`}
                  style={{
                    width: `${Math.round(correctnessScore * 100)}%`,
                  }}
                />
              </div>
              <span className="text-sm font-medium">
                {Math.round(correctnessScore * 100)}%
              </span>
            </div>
          </div>
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger
                render={
                  <span className="inline-flex cursor-help text-zinc-400 hover:text-zinc-600">
                    <Icon icon={AlertCircleIcon} size={16} />
                  </span>
                }
              />
              <TooltipContent>
                <Text variant="small" tone="primary" className="max-w-xs">
                  AI-generated estimate of how well this execution achieved its
                  intended purpose. This score indicates
                  {correctnessScore >= 0.8
                    ? " the agent was highly successful."
                    : correctnessScore >= 0.6
                      ? " the agent was mostly successful with minor issues."
                      : correctnessScore >= 0.4
                        ? " the agent was partially successful with some gaps."
                        : " the agent had limited success with significant issues."}
                </Text>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
      )}
    </div>
  );
}
