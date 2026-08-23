"use client";

import { cn } from "@/lib/utils";
import { StatusIcon } from "../TaskProgressBar/components/StatusIcon/StatusIcon";

type PhaseStatus = "pending" | "in_progress" | "completed";

interface Phase {
  content: string;
  status: PhaseStatus;
}

interface Props {
  phases: Phase[];
}

const STATUSES: PhaseStatus[] = ["pending", "in_progress", "completed"];

export function toPhases(value: unknown): Phase[] {
  if (!Array.isArray(value)) return [];
  return value.flatMap((entry) => {
    if (!entry || typeof entry !== "object") return [];
    const { content, status } = entry as Record<string, unknown>;
    if (typeof content !== "string" || !content.trim()) return [];
    return [
      {
        content: content.trim(),
        status: STATUSES.includes(status as PhaseStatus)
          ? (status as PhaseStatus)
          : "pending",
      },
    ];
  });
}

/** The delegated run's plan as a timeline — what is done, what is happening
 *  now, what is still queued. Elapsed seconds alone say a run is alive but
 *  not what it is doing; this is the part the user can actually read. */
export function SubSessionPhases({ phases }: Props) {
  if (phases.length === 0) return null;
  const completed = phases.filter((p) => p.status === "completed").length;

  return (
    <div className="mt-2 border-t border-zinc-100 pl-1 pt-2.5">
      <p className="mb-1.5 text-[11px] uppercase tracking-wide text-zinc-400">
        {completed} of {phases.length} steps done
      </p>
      <ul className="flex flex-col gap-1">
        {phases.map((phase, i) => (
          <li
            key={`${i}-${phase.content}`}
            className="flex items-center gap-2 text-xs"
          >
            <span className="flex size-4 shrink-0 items-center justify-center">
              <StatusIcon status={phase.status} />
            </span>
            <span
              className={cn(
                "truncate",
                phase.status === "completed" && "text-zinc-400 line-through",
                phase.status === "in_progress" && "font-medium text-zinc-700",
                phase.status === "pending" && "text-zinc-500",
              )}
            >
              {phase.content}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}
