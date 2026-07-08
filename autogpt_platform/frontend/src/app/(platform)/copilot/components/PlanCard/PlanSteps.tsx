import { cn } from "@/lib/utils";
import { TargetIcon, WrenchIcon } from "@phosphor-icons/react";
import type { PlanStep } from "./helpers";

interface Props {
  steps: PlanStep[];
}

export function PlanSteps({ steps }: Props) {
  if (steps.length === 0) return null;

  return (
    <ol className="flex flex-col gap-3">
      {steps.map((step, index) => (
        <li key={step.id || index} className="flex gap-3">
          <span className="mt-0.5 flex size-5 shrink-0 items-center justify-center rounded-full bg-zinc-200 text-xs font-medium text-zinc-700">
            {index + 1}
          </span>
          <div className="min-w-0 flex-1">
            <p className="text-sm text-zinc-800">{step.description}</p>
            {step.successCriteria && (
              <p className="mt-1 flex items-start gap-1 text-xs text-zinc-500">
                <TargetIcon className="mt-0.5 size-3 shrink-0" />
                <span>{step.successCriteria}</span>
              </p>
            )}
            {step.expectedTools.length > 0 && (
              <div className="mt-1.5 flex flex-wrap items-center gap-1">
                <WrenchIcon className="size-3 shrink-0 text-zinc-400" />
                {step.expectedTools.map((tool) => (
                  <span
                    key={tool}
                    className={cn(
                      "rounded-md bg-zinc-100 px-1.5 py-0.5",
                      "font-mono text-[11px] text-zinc-600",
                    )}
                  >
                    {tool}
                  </span>
                ))}
              </div>
            )}
          </div>
        </li>
      ))}
    </ol>
  );
}
