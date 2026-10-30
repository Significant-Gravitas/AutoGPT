import type { PlanProgress } from "@/app/api/__generated__/models/planProgress";
import { Text } from "@/components/atoms/Text/Text";
import { ChecklistMark } from "../../../../components/ChecklistMark";
import { getProgressSteps } from "./helpers";

interface Props {
  progress: PlanProgress;
}

// Stands where the heard card will be, and reports the server's own progress
// rather than a timer's.
export function PlanProgressCard({ progress }: Props) {
  return (
    <ul
      aria-label="Writing your plan"
      className="flex min-h-[5.375rem] flex-col justify-center gap-1.5 rounded-2xl border border-border-base bg-surface px-3.5 py-2.5"
    >
      {getProgressSteps(progress).map((step) => (
        <li key={step.label} className="flex items-center gap-2.5">
          <ChecklistMark key={step.state} state={step.state} />
          <Text
            variant={step.state === "active" ? "body-medium" : "body"}
            as="span"
            tone={step.state === "pending" ? "muted" : undefined}
          >
            {step.label}
          </Text>
        </li>
      ))}
    </ul>
  );
}
