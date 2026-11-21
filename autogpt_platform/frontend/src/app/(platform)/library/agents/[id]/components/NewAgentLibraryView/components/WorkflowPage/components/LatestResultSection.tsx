import { ChartBlockView } from "@/app/(platform)/copilot/tools/RenderUI/components/ChartBlockView";
import { StatsBlockView } from "@/app/(platform)/copilot/tools/RenderUI/components/StatsBlockView";
import { TableBlockView } from "@/app/(platform)/copilot/tools/RenderUI/components/TableBlockView";
import type { PageLatestResult } from "@/app/api/__generated__/models/pageLatestResult";
import { Text } from "@/components/atoms/Text/Text";
import { formatDate } from "@/lib/utils/time";
import { RunOutputs } from "../../selected-views/SelectedRunView/components/RunOutputs";
import { isRunActive } from "../helpers";

interface Props {
  section: PageLatestResult;
}

export function LatestResultSection({ section }: Props) {
  if (!section.run_id) {
    return (
      <Text variant="body" tone="secondary">
        No finished run yet. Run the workflow to see its result here.
      </Text>
    );
  }
  if (isRunActive(section.run_status)) {
    return (
      <Text variant="body" tone="secondary" aria-live="polite">
        Running… the result shows here when it finishes.
      </Text>
    );
  }
  const otherOutputs = section.other_outputs ?? {};
  return (
    <div className="flex min-w-0 flex-col gap-3">
      {section.finished_at ? (
        <Text variant="small" tone="secondary">
          Finished {formatDate(new Date(section.finished_at))}
        </Text>
      ) : null}
      {(section.blocks ?? []).map((block, index) => {
        switch (block.type) {
          case "stats":
            return <StatsBlockView key={index} block={block} />;
          case "table":
            return <TableBlockView key={index} block={block} />;
          case "chart":
            return <ChartBlockView key={index} block={block} />;
        }
      })}
      {Object.keys(otherOutputs).length ? (
        <RunOutputs outputs={otherOutputs as Record<string, unknown[]>} />
      ) : null}
    </div>
  );
}
