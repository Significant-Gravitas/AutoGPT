import { PauseCircleIcon } from "@hugeicons/core-free-icons";

import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";

interface Props {
  title: string;
  /** What became of the pause, when that is known. */
  note: string | null;
}

export function PausedLine({ title, note }: Props) {
  return (
    <div
      role="note"
      aria-label={note ? `${title}. ${note}` : title}
      className="my-2 flex flex-wrap items-center gap-x-2 gap-y-1 rounded-xl bg-surface-muted px-3 py-2"
    >
      <Icon
        icon={PauseCircleIcon}
        size={14}
        className="shrink-0 text-foreground-subtle"
      />
      <Text variant="small" className="!text-foreground-secondary">
        {title}
      </Text>
      {note && (
        <Text variant="small" className="!text-foreground-subtle">
          {note}
        </Text>
      )}
    </div>
  );
}
