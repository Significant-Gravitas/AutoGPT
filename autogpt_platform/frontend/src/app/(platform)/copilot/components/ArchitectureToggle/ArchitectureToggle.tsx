"use client";

import { Switch } from "@/components/atoms/Switch/Switch";
import { toast } from "@/components/molecules/Toast/use-toast";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import { Flag, useGetFlag } from "@/services/feature-flags/use-get-flag";
import { StrategyIcon } from "@phosphor-icons/react";
import { useCopilotUIStore } from "../../store";

interface Props {
  className?: string;
}

/**
 * Copilot-scoped switch that forces the two-phase planner/executor
 * architecture on or off for subsequent messages. On = the backend plans
 * the task then runs an executor loop (still only when the request looks
 * multi-step); off = the classic single-model loop. Gated behind the
 * ``copilot-planner-executor`` flag so it only surfaces for the test cohort.
 */
export function ArchitectureToggle({ className }: Props) {
  const isEnabled = useGetFlag(Flag.COPILOT_PLANNER_EXECUTOR);
  const isPlannerSplitEnabled = useCopilotUIStore(
    (s) => s.isPlannerSplitEnabled,
  );
  const setPlannerSplitEnabled = useCopilotUIStore(
    (s) => s.setPlannerSplitEnabled,
  );

  if (!isEnabled) return null;

  function handleChange(next: boolean) {
    setPlannerSplitEnabled(next);
    toast({
      title: next ? "Planner architecture on" : "Planner architecture off",
      description: next
        ? "Multi-step tasks are planned first, then run by an executor."
        : "Using the classic single-model loop.",
    });
  }

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <label
          className={cn(
            "flex cursor-pointer select-none items-center gap-2 rounded-full",
            "border border-zinc-200 bg-white/90 px-3 py-1.5 shadow-sm backdrop-blur",
            "transition-colors hover:bg-white",
            className,
          )}
        >
          <StrategyIcon
            className={cn(
              "size-4",
              isPlannerSplitEnabled ? "text-violet-600" : "text-zinc-400",
            )}
          />
          <span className="text-xs font-medium text-zinc-700">Planner</span>
          <Switch
            checked={isPlannerSplitEnabled}
            onCheckedChange={handleChange}
            aria-label="Toggle planner/executor architecture"
          />
        </label>
      </TooltipTrigger>
      <TooltipContent side="bottom" className="max-w-56">
        When on, multi-step tasks are planned by a planner model and carried
        out by an executor model. When off, the classic single-model loop is
        used.
      </TooltipContent>
    </Tooltip>
  );
}
