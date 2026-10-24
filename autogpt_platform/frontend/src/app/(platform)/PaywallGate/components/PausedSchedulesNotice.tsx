import { Icon } from "@/components/atoms/Icon/Icon";
import { PauseCircleIcon } from "@hugeicons/core-free-icons";
import { pausedSchedulesText } from "../helpers";

interface Props {
  count: number;
}

export function PausedSchedulesNotice({ count }: Props) {
  if (count <= 0) return null;
  return (
    <div className="inline-flex items-center gap-2 rounded-full border border-amber-200 bg-amber-50/70 px-4 py-1.5 text-sm text-zinc-700">
      <Icon icon={PauseCircleIcon} className="size-5 shrink-0 text-amber-600" />
      <span>{pausedSchedulesText(count)}</span>
    </div>
  );
}
