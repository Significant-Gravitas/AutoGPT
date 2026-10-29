"use client";

import type { GraphExecution } from "@/app/api/__generated__/models/graphExecution";
import type { GraphExecutionWithNodes } from "@/app/api/__generated__/models/graphExecutionWithNodes";
import { Icon } from "@/components/atoms/Icon/Icon";
import { CircleAlertGlyph } from "@/components/atoms/Icon/drawnIcons";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/atoms/Tooltip/BaseTooltip";

interface Props {
  run: GraphExecution | GraphExecutionWithNodes;
}

export function RunSummary({ run }: Props) {
  if (!run.stats?.activity_status) return null;

  const correctnessScore = run.stats.correctness_score;

  return (
    <div className="space-y-4">
      <p className="text-sm leading-relaxed text-foreground-body">
        {run.stats.activity_status}
      </p>

      {typeof correctnessScore === "number" && (
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium text-foreground-secondary">
              Success Estimate:
            </span>
            <div className="flex items-center gap-2">
              <div className="relative h-2 w-16 overflow-hidden rounded-full bg-surface-active">
                <div
                  className={`h-full transition-all ${
                    correctnessScore >= 0.8
                      ? "bg-success"
                      : correctnessScore >= 0.6
                        ? "bg-yellow-500"
                        : correctnessScore >= 0.4
                          ? "bg-orange-500"
                          : "bg-error"
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
                <Icon
                  icon={CircleAlertGlyph}
                  aria-label="Icon"
                  className="size-4 cursor-help text-foreground-subtle hover:text-foreground-secondary"
                />
              </TooltipTrigger>
              <TooltipContent>
                <p className="max-w-xs">
                  AI-generated estimate of how well this execution achieved its
                  intended purpose. This score indicates
                  {correctnessScore >= 0.8
                    ? " the workflow was highly successful."
                    : correctnessScore >= 0.6
                      ? " the workflow was mostly successful with minor issues."
                      : correctnessScore >= 0.4
                        ? " the workflow was partially successful with some gaps."
                        : " the workflow had limited success with significant issues."}
                </p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
      )}
    </div>
  );
}
