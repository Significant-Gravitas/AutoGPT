"use client";

import { Button } from "@/components/atoms/Button/Button";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { useScheduleDetailHeader } from "../../../RunDetailHeader/useScheduleDetailHeader";
import { Delete02Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

type Props = {
  agent: LibraryAgent;
  scheduleId: string;
  onDeleted?: () => void;
};

export function DeleteScheduleButton({ agent, scheduleId, onDeleted }: Props) {
  const { deleteSchedule, isDeleting } = useScheduleDetailHeader(
    agent.graph_id,
    scheduleId,
    agent.graph_version,
  );

  return (
    <Button
      variant="secondary"
      size="small"
      onClick={() => {
        deleteSchedule();
        if (onDeleted) onDeleted();
      }}
      loading={isDeleting}
    >
      <Icon icon={Delete02Icon} size={16} /> Delete schedule
    </Button>
  );
}
