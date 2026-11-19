"use client";

import type { GraphExecutionMeta } from "@/app/api/__generated__/models/graphExecutionMeta";
import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Calendar03Icon } from "@hugeicons/core-free-icons";
import { useState } from "react";
import { RunNowButton } from "./RunNowButton";
import { SchedulePanel } from "./SchedulePanel";

interface Props {
  graphID: string;
  title: string;
  agent: LibraryAgent | undefined;
  run: GraphExecutionMeta | undefined;
  onStarted: (executionID: string) => void;
  onSetup: () => void;
}

/** The tab's head, about the workflow itself: its name, Run now and
 *  Schedule. What one run did sits with that run in the list below. */
export function RunHeader({
  graphID,
  title,
  agent,
  run,
  onStarted,
  onSetup,
}: Props) {
  const [isScheduling, setIsScheduling] = useState(false);
  return (
    <div className="flex flex-col gap-3 px-1">
      <div className="flex flex-col gap-1.5">
        <Text variant="body-md-medium" as="h2" tone="ink" className="truncate">
          {title}
        </Text>
      </div>
      <div className="flex flex-wrap items-center gap-1.5">
        <RunNowButton agent={agent} onStarted={onStarted} onSetup={onSetup} />
        <Button
          variant="secondary"
          size="xs"
          leadingIcon={Calendar03Icon}
          aria-expanded={isScheduling}
          onClick={() => setIsScheduling(!isScheduling)}
        >
          Schedule
        </Button>
      </div>
      {isScheduling ? (
        <SchedulePanel
          graphID={graphID}
          agent={agent}
          run={run}
          onClose={() => setIsScheduling(false)}
        />
      ) : null}
    </div>
  );
}
