import type { PageSchedule } from "@/app/api/__generated__/models/pageSchedule";
import { Text } from "@/components/atoms/Text/Text";
import { formatDate } from "@/lib/utils/time";

interface Props {
  section: PageSchedule;
}

export function ScheduleSection({ section }: Props) {
  if (section.unavailable) {
    return (
      <Text variant="body" tone="secondary">
        The schedule could not be loaded right now.
      </Text>
    );
  }
  if (!section.is_scheduled) {
    return (
      <Text variant="body" tone="secondary">
        This workflow is not scheduled. Add a schedule from the Scheduled tab.
      </Text>
    );
  }
  return (
    <Text variant="body">
      {section.next_run
        ? `Runs on a schedule. Next run: ${formatDate(new Date(section.next_run))}`
        : "Runs on a schedule."}
    </Text>
  );
}
