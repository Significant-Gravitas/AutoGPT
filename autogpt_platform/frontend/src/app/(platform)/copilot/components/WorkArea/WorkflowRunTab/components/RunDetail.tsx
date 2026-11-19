"use client";

import {
  RunScreen,
  RunStepView,
} from "@/app/(platform)/runs/components/RunScreen";
import { runHref } from "@/app/(platform)/runs/helpers";
import { useRunControls } from "@/app/(platform)/runs/useRunControls";
import type { GraphExecution } from "@/app/api/__generated__/models/graphExecution";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { Button } from "@/components/atoms/Button/Button";
import {
  ArrowLeft01Icon,
  ArrowUpRight01Icon,
} from "@hugeicons/core-free-icons";
import { useState } from "react";
import type { Trace } from "../trace";

interface Props {
  graphID: string;
  title: string;
  runID: string;
  run: GraphExecution | undefined;
  trace: Trace;
  isLive: boolean;
  now: number;
  agent: LibraryAgent | undefined;
  onBack: () => void;
  onStarted: (executionID: string) => void;
  onSetup: () => void;
}

/** One run beside the chat, on the Affect UI Run Detail block. A block run
 *  opens as a page of its own in the tab, with a way back to the run; the
 *  run also opens as a full page. */
export function RunDetail({
  graphID,
  title,
  runID,
  run,
  trace,
  isLive,
  now,
  agent,
  onBack,
  onStarted,
  onSetup,
}: Props) {
  const [stepID, setStepID] = useState<string | null>(null);
  const controls = useRunControls({
    graphID,
    runID,
    agent,
    onStarted,
    onSetup,
  });
  const data = { graphID, title, run, trace, isLive, now };

  if (stepID) {
    return (
      <div aria-label={`Block run of ${title}`} className="flex flex-col gap-2">
        <div className="flex px-1">
          <Button
            variant="ghost-muted"
            size="xs"
            leadingIcon={ArrowLeft01Icon}
            className="-ml-1.5"
            onClick={() => setStepID(null)}
          >
            Back to the run
          </Button>
        </div>
        <RunStepView {...data} stepID={stepID} embed />
      </div>
    );
  }

  return (
    <div aria-label={`Run of ${title}`} className="flex flex-col gap-2">
      <div className="flex items-center justify-between px-1">
        <Button
          variant="ghost-muted"
          size="xs"
          leadingIcon={ArrowLeft01Icon}
          className="-ml-1.5"
          onClick={onBack}
        >
          All runs
        </Button>
        <Button
          as="NextLink"
          href={runHref(graphID, runID)}
          variant="ghost-muted"
          size="xs"
          leadingIcon={ArrowUpRight01Icon}
        >
          Open full page
        </Button>
      </div>
      <RunScreen
        {...data}
        opened={null}
        onOpen={setStepID}
        onRerun={controls.rerun}
        onStop={controls.stop}
        embed
      />
    </div>
  );
}
