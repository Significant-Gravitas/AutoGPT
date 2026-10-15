import type { ExpertIdentityMap } from "@/app/(platform)/copilot/useExpertMap";
import { Text } from "@/components/atoms/Text/Text";
import type { ReactNode } from "react";
import { SCHEDULED, type ScheduleCardData } from "../../helpers";
import { ScheduleCard } from "../ScheduleCard/ScheduleCard";
import { StatusIcon } from "../StatusIcon/StatusIcon";

interface Props {
  schedules: ScheduleCardData[];
  expertsById: ExpertIdentityMap;
  /** Rendered above the cards, e.g. the new-schedule card. */
  leading?: ReactNode;
}

export function ScheduledColumn({ schedules, expertsById, leading }: Props) {
  return (
    <section
      aria-label="Scheduled"
      className="flex h-[calc(100dvh-8rem)] min-h-[32rem] min-w-[18rem] max-w-[26rem] flex-1 basis-0 flex-col rounded-xl bg-zinc-100/80"
    >
      <header className="flex h-11 items-center gap-2 pl-3.5 pr-2">
        <StatusIcon columnId={SCHEDULED} />
        <Text variant="body-medium" tone="primary">
          Scheduled
        </Text>
        <Text variant="body" tone="muted" className="tabular-nums">
          {schedules.length}
        </Text>
      </header>
      <div className="flex min-h-0 flex-1 flex-col gap-2 overflow-y-auto px-2 pb-2 scrollbar-thin scrollbar-track-transparent scrollbar-thumb-zinc-300">
        {leading}
        {schedules.map((schedule) => (
          <ScheduleCard
            key={schedule.id}
            schedule={schedule}
            expert={
              schedule.expertId ? expertsById.get(schedule.expertId) : undefined
            }
          />
        ))}
        {schedules.length === 0 && !leading && (
          <div className="flex items-center justify-center rounded-lg border border-dashed border-zinc-300 py-8">
            <Text variant="small" tone="muted">
              Nothing scheduled
            </Text>
          </div>
        )}
      </div>
    </section>
  );
}
