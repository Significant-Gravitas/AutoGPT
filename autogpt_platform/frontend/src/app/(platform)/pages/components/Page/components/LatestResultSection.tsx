import { RunOutputs } from "@/app/(platform)/library/agents/[id]/components/NewAgentLibraryView/components/selected-views/SelectedRunView/components/RunOutputs";
import { usePostV1ExecuteGraphAgent } from "@/app/api/__generated__/endpoints/graphs/graphs";
import type { PageLatestResult } from "@/app/api/__generated__/models/pageLatestResult";
import type { RowRun } from "@/app/api/__generated__/models/rowRun";
import { okData } from "@/app/api/helpers";
import { Text } from "@/components/atoms/Text/Text";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { formatDate } from "@/lib/utils/time";
import { trackPageRunStarted } from "@/services/pages/pages-analytics";
import { useState } from "react";
import { isRunActive } from "../helpers";
import type { PageLive } from "../Page";
import { BlockView } from "./CardsSection";

interface Props {
  section: PageLatestResult;
  live?: PageLive;
}

export function LatestResultSection({ section, live }: Props) {
  const rowRun = useRowRun(live);
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
  const rowRuns = section.row_runs ?? [];
  return (
    <div className="flex min-w-0 flex-col gap-3">
      {section.finished_at ? (
        <Text variant="small" tone="secondary">
          Finished {formatDate(new Date(section.finished_at))}
        </Text>
      ) : null}
      {(section.blocks ?? []).map((block, index) => {
        const action = rowRuns.find((r) => r.block_index === index);
        return (
          <BlockView
            key={index}
            block={block}
            rowAction={
              action && live
                ? {
                    label: action.label,
                    pendingRow: rowRun.pending?.startsWith(`${index}:`)
                      ? Number(rowRun.pending.split(":")[1])
                      : null,
                    onRun: (row) => rowRun.start(action, index, row),
                  }
                : undefined
            }
          />
        );
      })}
      {Object.keys(otherOutputs).length ? (
        // RunOutputs pins its copy and download actions to this box's corner.
        <div className="relative">
          <RunOutputs outputs={otherOutputs as Record<string, unknown[]>} />
        </div>
      ) : null}
    </div>
  );
}

function useRowRun(live?: PageLive) {
  const [pending, setPending] = useState<string | null>(null);
  const { toast } = useToast();
  const execute = usePostV1ExecuteGraphAgent();

  async function start(action: RowRun, blockIndex: number, row: number) {
    if (!live) return;
    setPending(`${blockIndex}:${row}`);
    try {
      const response = await execute.mutateAsync({
        graphId: action.graph_id,
        graphVersion: action.graph_version,
        data: { inputs: { [action.input]: action.values[row] } },
      });
      const run = okData(response);
      if (!run) throw new Error("The run did not start");
      trackPageRunStarted("row_button");
      live.onRunStarted(action.library_agent_id, run.id);
    } catch {
      toast({ title: "Couldn't start the run", variant: "destructive" });
    } finally {
      setPending(null);
    }
  }

  return { pending, start };
}
