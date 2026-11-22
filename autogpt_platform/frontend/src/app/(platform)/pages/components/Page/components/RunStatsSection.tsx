import { ChartBlockView } from "@/app/(platform)/copilot/tools/RenderUI/components/ChartBlockView";
import { StatsBlockView } from "@/app/(platform)/copilot/tools/RenderUI/components/StatsBlockView";
import type { PageRunNumbers } from "@/app/api/__generated__/models/pageRunNumbers";
import type { PageRunTrend } from "@/app/api/__generated__/models/pageRunTrend";
import { Text } from "@/components/atoms/Text/Text";

const PERIODS: Record<string, string> = {
  today: "Today",
  "7d": "Last 7 days",
  "30d": "Last 30 days",
};

interface Props {
  section: PageRunNumbers | PageRunTrend;
}

export function RunStatsSection({ section }: Props) {
  if (section.kind === "run_numbers") {
    return (
      <div className="flex flex-col gap-2">
        <Text variant="small" tone="secondary">
          {PERIODS[section.period] ?? section.period}
        </Text>
        <StatsBlockView block={section.block} />
      </div>
    );
  }
  if (!section.block) {
    return (
      <Text variant="body" tone="secondary">
        The trend shows once the workflow has run twice.
      </Text>
    );
  }
  return <ChartBlockView block={section.block} />;
}
