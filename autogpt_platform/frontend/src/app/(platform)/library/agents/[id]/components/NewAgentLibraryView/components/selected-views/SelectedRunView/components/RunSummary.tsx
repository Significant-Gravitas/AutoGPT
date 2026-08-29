"use client";

import type { GetV1GetExecutionDetails200 } from "@/app/api/__generated__/models/getV1GetExecutionDetails200";
import { IconCircleAlert } from "@/components/__legacy__/ui/icons";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";

interface Props {
  run: GetV1GetExecutionDetails200;
}

export function RunSummary({ run }: Props) {
  if (!run.stats?.activity_status) return null;

  const correctnessScore = run.stats.correctness_score;

  return (
    <div className="space-y-4">
      <p className="text-sm leading-relaxed text-neutral-700">
        {run.stats.activity_status}
      </p>

      {typeof correctnessScore === "number" && (
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium text-neutral-600">
              Success Estimate:
            </span>
            <div className="flex items-center gap-2">
              <div className="relative h-2 w-16 overflow-hidden rounded-full bg-neutral-200">
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
              <TooltipTrigger asChild>
                <IconCircleAlert className="size-4 cursor-help text-neutral-400 hover:text-neutral-600" />
              </TooltipTrigger>
              <TooltipContent>
                <p className="max-w-xs">
                  AI-generated estimate of how well this execution achieved its
                  intended purpose. This score indicates
                  {correctnessScore >= 0.8
                    ? " the agent was highly successful."
                    : correctnessScore >= 0.6
                      ? " the agent was mostly successful with minor issues."
                      : correctnessScore >= 0.4
                        ? " the agent was partially successful with some gaps."
                        : " the agent had limited success with significant issues."}
                </p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
      )}
    </div>
  );
}
