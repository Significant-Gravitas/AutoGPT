import {
  getV1GetExecutionDetails,
  usePostV1ExecuteGraphAgent,
} from "@/app/api/__generated__/endpoints/graphs/graphs";
import type { PageButton } from "@/app/api/__generated__/models/pageButton";
import { okData } from "@/app/api/helpers";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { trackPageRunStarted } from "@/services/pages/pages-analytics";
import { useState } from "react";
import type { TableRow } from "../CollectionTable/helpers";
import { runInputs } from "./helpers";

const POLL_MS = 2_000;
const MAX_POLLS = 150;
const FINISHED = new Set(["COMPLETED", "FAILED", "TERMINATED"]);

interface Args {
  onRunStarted?(libraryAgentID: string, runID: string): void;
  /** The run wrote its result back: read the rows again. */
  onRunFinished(): void;
}

export function useRowButtons({ onRunStarted, onRunFinished }: Args) {
  const [running, setRunning] = useState<string | null>(null);
  const { toast } = useToast();
  const execute = usePostV1ExecuteGraphAgent();

  async function run(button: PageButton, row: TableRow) {
    setRunning(`${button.label}:${row.id}`);
    try {
      const response = await execute.mutateAsync({
        graphId: button.graph_id,
        graphVersion: button.graph_version,
        data: { inputs: runInputs(button, row) },
      });
      const started = okData(response);
      if (!started) throw new Error("The run did not start");
      trackPageRunStarted("row_button");
      onRunStarted?.(button.library_agent_id, started.id);
      await untilFinished(button.graph_id, started.id);
      onRunFinished();
    } catch {
      toast({ title: "Couldn't start the run", variant: "destructive" });
    } finally {
      setRunning(null);
    }
  }

  return {
    run,
    isRunning: (button: PageButton, row: TableRow) =>
      running === `${button.label}:${row.id}`,
    isBusy: running !== null,
  };
}

async function untilFinished(graphID: string, runID: string) {
  for (let poll = 0; poll < MAX_POLLS; poll++) {
    await new Promise((resolve) => setTimeout(resolve, POLL_MS));
    const details = okData(await getV1GetExecutionDetails(graphID, runID));
    if (details && FINISHED.has(details.status)) return;
  }
}
