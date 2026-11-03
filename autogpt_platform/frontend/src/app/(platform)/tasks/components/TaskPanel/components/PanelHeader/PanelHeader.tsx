import { Cancel01Icon } from "@hugeicons/core-free-icons";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { taskKey } from "../../../../data/helpers";
import type { TaskDetail } from "../../../../data/types";

interface Props {
  task: TaskDetail | undefined;
  onClose(): void;
}

export function PanelHeader({ task, onClose }: Props) {
  return (
    <div className="sticky top-0 z-10 flex h-12 shrink-0 items-center gap-2 bg-surface pr-2 pl-5 shadow-hairline-b">
      {task ? (
        <>
          <Text
            as="span"
            variant="small-mono"
            tone="faint"
            className="shrink-0"
          >
            {taskKey(task.number)}
          </Text>
          <Text
            as="span"
            variant="inbox-medium"
            tone="ink"
            className="min-w-0 flex-1 truncate"
          >
            {task.title}
          </Text>
        </>
      ) : (
        <span className="flex-1" />
      )}
      <Button
        variant="ghost-muted"
        size="icon-xs"
        aria-label="Close task"
        onClick={onClose}
      >
        <Icon icon={Cancel01Icon} size={14} />
      </Button>
    </div>
  );
}
