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
      <p
        // TODO(tokens): no exact role for text-neutral-700 (#404040)
        className="text-sm leading-relaxed text-neutral-700"
      >
        {run.stats.activity_status}
      </p>

      {typeof correctnessScore === "number" && (
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <span
              // TODO(tokens): no exact role for text-neutral-600 (#525252)
              className="text-sm font-medium text-neutral-600"
            >
              Success Estimate:
            </span>
            <div className="flex items-center gap-2">
              <div
                // TODO(tokens): no exact role for bg-neutral-200 (#E5E5E5)
                className="relative h-2 w-16 overflow-hidden rounded-full bg-neutral-200"
              >
                <div
                  className={`h-full transition-all ${
                    correctnessScore >= 0.8
                      ? // TODO(tokens): no exact role for bg-green-500 (#16A34A)
                        "bg-green-500"
                      : correctnessScore >= 0.6
                        ? // TODO(tokens): no exact role for bg-yellow-500 (#F5C000)
                          "bg-yellow-500"
                        : correctnessScore >= 0.4
                          ? // TODO(tokens): no exact role for bg-orange-500 (#FE8700)
                            "bg-orange-500"
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
                  // TODO(tokens): no exact role for text-neutral-400 (#A3A3A3), hover:text-neutral-600 (#525252)
                  className="size-4 cursor-help text-neutral-400 hover:text-neutral-600"
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
