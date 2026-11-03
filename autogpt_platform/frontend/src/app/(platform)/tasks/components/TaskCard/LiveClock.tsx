import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { Clock01Icon } from "@hugeicons/core-free-icons";
import { formatClock } from "./helpers";
import { useElapsed } from "./useElapsed";

interface Props {
  startedAt: string | null;
}

export function LiveClock({ startedAt }: Props) {
  const elapsed = useElapsed(startedAt);
  if (elapsed === null) return null;
  return (
    <span className="inline-flex items-center gap-1" title="Working for">
      <Icon icon={Clock01Icon} size={12} tone="info" />
      <Text variant="small-mono" tone="info-strong">
        {formatClock(elapsed)}
      </Text>
    </span>
  );
}
