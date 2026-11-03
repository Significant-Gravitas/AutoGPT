import type { TaskPlanStep } from "@/app/(platform)/tasks/data/types";
import { Text } from "@/components/atoms/Text/Text";
import { AssigneeAvatar } from "./AssigneeAvatar";

interface Props {
  step: TaskPlanStep;
  index: number;
}

export function PlanStepRow({ step, index }: Props) {
  return (
    <li className="flex items-start gap-3 py-2">
      <Text
        variant="small-medium"
        as="span"
        tone="muted"
        tabular
        className="mt-0.5 w-4 shrink-0 text-right"
      >
        {index + 1}
      </Text>
      <div className="min-w-0 flex-1">
        <Text variant="body-medium" tone="primary" className="text-pretty">
          {step.title}
        </Text>
        {step.description ? (
          <Text variant="small" tone="secondary" className="text-pretty">
            {step.description}
          </Text>
        ) : null}
      </div>
      {step.expert ? (
        <div className="flex shrink-0 items-center gap-1.5">
          <AssigneeAvatar assignee={step.expert} size={20} />
          <Text variant="small-medium" as="span" tone="body">
            {step.expert.name}
          </Text>
        </div>
      ) : null}
    </li>
  );
}
