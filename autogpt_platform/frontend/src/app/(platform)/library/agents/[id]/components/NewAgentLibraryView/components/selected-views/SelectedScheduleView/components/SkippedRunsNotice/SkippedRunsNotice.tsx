import type { RunsSkipped } from "@/app/api/__generated__/models/runsSkipped";
import { Text } from "@/components/atoms/Text/Text";
import {
  Alert,
  AlertDescription,
  AlertTitle,
} from "@/components/molecules/Alert/Alert";
import { formatInTimezone } from "@/lib/timezone-utils";
import { SKIPPED_RUNS_COPY } from "./helpers";

interface Props {
  runsSkipped: RunsSkipped;
  timezone: string;
}

export function SkippedRunsNotice({ runsSkipped, timezone }: Props) {
  const { reason, fix } = SKIPPED_RUNS_COPY[runsSkipped.problem];
  const since = formatInTimezone(runsSkipped.since, timezone, {
    year: "numeric",
    month: "long",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  });

  return (
    <Alert variant="warning">
      <AlertTitle>This schedule is skipping its runs</AlertTitle>
      <AlertDescription>
        <div className="flex flex-col gap-1">
          <Text variant="body">
            {reason} {fix}
          </Text>
          <Text variant="small" tone="secondary">
            Skipping since {since}
          </Text>
        </div>
      </AlertDescription>
    </Alert>
  );
}
