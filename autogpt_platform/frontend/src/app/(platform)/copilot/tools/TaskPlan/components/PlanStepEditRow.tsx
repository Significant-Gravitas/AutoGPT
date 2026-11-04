import { Cancel01Icon } from "@hugeicons/core-free-icons";
import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Text } from "@/components/atoms/Text/Text";
import type { TaskPlanStep } from "@/app/(platform)/tasks/data/types";

interface Props {
  planId: string;
  step: TaskPlanStep;
  index: number;
  canRemove: boolean;
  onTitleChange: (index: number, title: string) => void;
  onRemove: (index: number) => void;
}

export function PlanStepEditRow({
  planId,
  step,
  index,
  canRemove,
  onTitleChange,
  onRemove,
}: Props) {
  return (
    <li className="flex items-center gap-3 py-2">
      <span className="w-4 shrink-0 text-right">
        <Text variant="inbox" as="span" tone="faint" tabular>
          {index + 1}
        </Text>
      </span>
      <div className="min-w-0 flex-1">
        <Input
          id={`plan-${planId}-step-${index}`}
          label={`Step ${index + 1}`}
          hideLabel
          size="small"
          value={step.title}
          onChange={(e) => onTitleChange(index, e.target.value)}
        />
      </div>
      <Button
        variant="ghost-danger"
        size="icon-xs"
        leadingIcon={Cancel01Icon}
        aria-label={`Remove step ${index + 1}`}
        disabled={!canRemove}
        onClick={() => onRemove(index)}
      />
    </li>
  );
}
