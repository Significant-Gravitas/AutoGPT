import type { TaskPlanStep } from "@/app/(platform)/tasks/data/types";
import { Text } from "@/components/atoms/Text/Text";
import { AssigneeAvatar } from "./AssigneeAvatar";

interface Props {
  step: TaskPlanStep;
  index: number;
}

export function PlanStepRow({ step, index }: Props) {
  return (
    <li className="flex items-start gap-3 py-2.5">
      <span className="w-4 shrink-0 text-right">
        <Text variant="inbox" as="span" tone="faint" tabular>
          {index + 1}
        </Text>
      </span>
      <div className="min-w-0 flex-1">
        <Text variant="inbox-medium" tone="ink" className="text-pretty">
          {step.title}
        </Text>
        {step.description ? (
          <Text variant="inbox" tone="text" className="text-pretty">
            {step.description}
          </Text>
        ) : null}
      </div>
      {step.expert ? (
        <div className="flex shrink-0 items-center gap-1.5">
          <AssigneeAvatar assignee={step.expert} size={20} />
          <Text variant="inbox" as="span" tone="text">
            {step.expert.name}
          </Text>
        </div>
      ) : null}
    </li>
  );
}
