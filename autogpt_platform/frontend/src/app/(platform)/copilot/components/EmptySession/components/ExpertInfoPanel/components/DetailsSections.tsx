import type { Expert } from "@/app/api/__generated__/models/expert";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { Calendar03Icon } from "@hugeicons/core-free-icons";
import { getWorkInfo, type PanelSchedule } from "../helpers";
import {
  InfoEmpty,
  InfoGroup,
  InfoList,
  InfoLoading,
  InfoMeta,
  InfoPill,
  InfoRow,
} from "./InfoParts";

interface SchedulesProps {
  schedules: PanelSchedule[];
  isPaused: boolean;
  isLoading: boolean;
}

export function ScheduledGroup({
  schedules,
  isPaused,
  isLoading,
}: SchedulesProps) {
  return (
    <InfoGroup label="Scheduled" count={schedules.length}>
      {isLoading ? (
        <InfoLoading />
      ) : schedules.length === 0 ? (
        <InfoEmpty>Nothing scheduled yet.</InfoEmpty>
      ) : (
        <InfoList>
          {schedules.map((schedule) => (
            <InfoRow
              key={schedule.id}
              tile={<Icon icon={Calendar03Icon} size={14} />}
              title={schedule.title}
              subtitle={schedule.cadence}
              trailing={
                isPaused ? (
                  <InfoPill>Paused</InfoPill>
                ) : schedule.nextRun ? (
                  <InfoMeta>{schedule.nextRun}</InfoMeta>
                ) : null
              }
            />
          ))}
        </InfoList>
      )}
    </InfoGroup>
  );
}

export function WorkInfoGroup({ expert }: { expert: Expert }) {
  return (
    <InfoGroup label="Work info">
      <InfoList>
        {getWorkInfo(expert).map((row) => (
          <InfoRow
            key={row.label}
            tile={<Icon icon={row.icon} size={14} />}
            title={row.label}
            trailing={
              <Text variant="inbox" as="span" tone="text" className="shrink-0">
                {row.value}
              </Text>
            }
          />
        ))}
      </InfoList>
    </InfoGroup>
  );
}
