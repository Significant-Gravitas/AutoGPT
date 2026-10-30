import { Icon } from "@/components/atoms/Icon/Icon";
import { PauseCircleIcon } from "@hugeicons/core-free-icons";
import { pausedSchedulesText } from "../helpers";

interface Props {
  count: number;
}

export function PausedSchedulesNotice({ count }: Props) {
  if (count <= 0) return null;
  return (
    <div className="inline-flex items-center gap-2 rounded-full border border-warning-border bg-warning-surface/70 px-4 py-1.5 text-sm text-foreground-body">
      <Icon
        icon={PauseCircleIcon}
        className="size-5 shrink-0 text-warning-foreground"
      />
      <span>{pausedSchedulesText(count)}</span>
    </div>
  );
}
