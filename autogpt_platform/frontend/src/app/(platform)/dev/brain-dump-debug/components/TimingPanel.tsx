import { Badge } from "@/components/atoms/Badge/Badge";
import { Text } from "@/components/atoms/Text/Text";
import { Timer01Icon } from "@hugeicons/core-free-icons";
import {
  BUDGET_DUMP_LENGTH,
  EXTRACT_BUDGET_MS,
  formatMs,
  TRANSCRIBE_BUDGET_MS,
} from "../helpers";
import { isOverBudget, type WaterfallStage } from "../waterfall";
import { DebugNote, DebugPanel } from "./DebugPanel";

interface Props {
  stages: WaterfallStage[];
  finalizeRoundTripMs: number | null;
}

export function TimingPanel({ stages, finalizeRoundTripMs }: Props) {
  return (
    <DebugPanel
      title="Timing waterfall"
      description={`Budget for a ${BUDGET_DUMP_LENGTH}: transcription within ${formatMs(TRANSCRIBE_BUDGET_MS)} of pressing Done, extraction within ${formatMs(EXTRACT_BUDGET_MS)} after that.`}
      icon={Timer01Icon}
      action={
        finalizeRoundTripMs !== null ? (
          <Badge variant="info">
            finalize round-trip {formatMs(finalizeRoundTripMs)}
          </Badge>
        ) : undefined
      }
    >
      <div className="flex flex-col gap-3">
        {stages.map((stage) => (
          <StageRow key={stage.id} stage={stage} />
        ))}
      </div>
      <div className="mt-6">
        <DebugNote>
          Finalize is a single synchronous call that assembles, scans, stores,
          transcribes and extracts, so the round-trip above is the whole
          post-Done budget. Per-phase numbers come from polled status
          transitions and are only available while this page is open across a
          run.
        </DebugNote>
      </div>
    </DebugPanel>
  );
}

function StageRow({ stage }: { stage: WaterfallStage }) {
  const measured = stage.durationMs !== null;
  const exceeded = isOverBudget(stage);

  return (
    <div className="flex flex-col gap-2 rounded-large border border-zinc-200 px-4 py-3 sm:flex-row sm:items-center sm:justify-between">
      <div className="flex flex-col gap-1">
        <div className="flex items-center gap-3">
          <Text variant="body-medium">{stage.label}</Text>
          {measured ? (
            <Badge variant={exceeded ? "error" : "success"}>
              {exceeded ? "over budget" : "measured"}
            </Badge>
          ) : (
            <Badge variant="info">unmeasured</Badge>
          )}
        </div>
        <Text variant="small" className="max-w-prose text-zinc-500">
          {stage.source}
        </Text>
      </div>
      <div className="flex shrink-0 flex-col sm:items-end">
        <Text variant="large-semibold" className="font-mono">
          {formatMs(stage.durationMs)}
        </Text>
        <Text variant="small" className="text-zinc-500">
          {stage.budgetMs === null
            ? "no budget"
            : `budget ${formatMs(stage.budgetMs)}`}
        </Text>
      </div>
    </div>
  );
}
