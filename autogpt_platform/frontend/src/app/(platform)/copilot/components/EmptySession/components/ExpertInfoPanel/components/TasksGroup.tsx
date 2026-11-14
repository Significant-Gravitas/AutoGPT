import type { Task } from "@/app/api/__generated__/models/task";
import { BOARD_HREF, statusOf, taskHref } from "@/app/(platform)/tasks/helpers";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { formatTimeAgo } from "@/lib/utils/time";
import { CheckListIcon } from "@hugeicons/core-free-icons";
import Link from "next/link";
import {
  FOCUS,
  InfoEmpty,
  InfoGroup,
  InfoList,
  InfoLoading,
  InfoPill,
  InfoRow,
} from "./InfoParts";

interface Props {
  tasks: Task[];
  isLoading: boolean;
  /** The whole team's tasks rather than one expert's: each row names who
   *  has it. */
  ownerName?: (task: Task) => string;
}

export function TasksGroup({ tasks, isLoading, ownerName }: Props) {
  const isTeam = Boolean(ownerName);
  return (
    <InfoGroup
      label={isTeam ? "Recent tasks" : "Tasks"}
      count={isTeam ? undefined : tasks.length}
      action={
        <Link
          href={BOARD_HREF}
          className={cn("rounded-md hover:underline", FOCUS)}
        >
          <Text variant="small" as="span" tone="quiet">
            View all
          </Text>
        </Link>
      }
    >
      {isLoading ? (
        <InfoLoading />
      ) : tasks.length === 0 ? (
        <InfoEmpty>
          {isTeam ? "No tasks yet." : "No tasks assigned yet."}
        </InfoEmpty>
      ) : (
        <InfoList>
          {tasks.map((task) => {
            const status = statusOf(task.status);
            return (
              <InfoRow
                key={task.id}
                href={taskHref(task) ?? undefined}
                tile={<Icon icon={CheckListIcon} size={14} />}
                title={task.title}
                subtitle={[
                  ownerName?.(task),
                  formatTimeAgo(String(task.updated_at)),
                ]
                  .filter(Boolean)
                  .join(" · ")}
                trailing={<InfoPill tone={status.tone}>{status.word}</InfoPill>}
              />
            );
          })}
        </InfoList>
      )}
    </InfoGroup>
  );
}
