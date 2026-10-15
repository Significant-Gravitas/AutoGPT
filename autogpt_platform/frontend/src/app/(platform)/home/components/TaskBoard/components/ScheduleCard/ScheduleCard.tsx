import type { ExpertIdentity } from "@/app/(platform)/copilot/useExpertMap";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { FlowIcon } from "@hugeicons/core-free-icons";
import Link from "next/link";
import {
  formatTimeUntil,
  SCHEDULED,
  type ScheduleCardData,
} from "../../helpers";
import { ExpertByline } from "../ExpertByline/ExpertByline";
import { StatusIcon } from "../StatusIcon/StatusIcon";

interface Props {
  schedule: ScheduleCardData;
  expert: ExpertIdentity | undefined;
}

export function ScheduleCard({ schedule, expert }: Props) {
  const isWorkflow = schedule.kind === "graph";
  const nextRun = formatTimeUntil(schedule.nextRunTime);

  return (
    <Link
      href={schedule.href}
      className="block rounded-lg border border-dashed border-zinc-300 bg-white/70 px-3.5 pb-3 pt-3 outline-none transition-colors hover:border-zinc-400 hover:bg-white focus-visible:ring-2 focus-visible:ring-violet-400"
    >
      {isWorkflow && !expert ? (
        <div className="flex items-center gap-1.5">
          <span className="flex size-5 items-center justify-center rounded-full border border-zinc-200 bg-white text-zinc-500">
            <Icon icon={FlowIcon} size={12} />
          </span>
          <Text variant="small" tone="muted">
            Workflow
          </Text>
        </div>
      ) : (
        <ExpertByline expert={expert} />
      )}
      <div className="mt-2.5 flex items-start gap-2">
        <StatusIcon columnId={SCHEDULED} className="mt-[0.2rem]" />
        <Text variant="body-medium" tone="primary" className="line-clamp-2">
          {schedule.title}
        </Text>
      </div>
      <div className="mt-2.5 flex items-center gap-3">
        <Text variant="small" tone="muted" className="truncate">
          {isWorkflow
            ? `Workflow · ${schedule.recurrence}`
            : schedule.recurrence}
        </Text>
        {nextRun && (
          <Text variant="small" tone="muted" className="ml-auto shrink-0">
            {nextRun}
          </Text>
        )}
      </div>
    </Link>
  );
}
