import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { Calendar03Icon } from "@hugeicons/core-free-icons";
import { formatDue, isOverdue } from "./helpers";

interface Props {
  due: string;
  isDone?: boolean;
}

export function DueChip({ due, isDone = false }: Props) {
  const overdue = !isDone && isOverdue(due);
  const tone = overdue ? "error" : "quiet";
  return (
    <span
      className="inline-flex items-center gap-1"
      title={overdue ? "Overdue" : "Due date"}
    >
      <Icon icon={Calendar03Icon} size={12} tone={tone} />
      <Text as="span" variant="small" tone={tone}>
        {formatDue(due)}
      </Text>
    </span>
  );
}
