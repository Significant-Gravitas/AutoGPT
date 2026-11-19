"use client";

import { useState } from "react";
import { Text } from "@/components/atoms/Text/Text";
import {
  InfoGroup,
  InfoList,
  InfoLoading,
} from "../../EmptySession/components/ExpertInfoPanel/components/InfoParts";
import { BlockNote } from "../blocks";
import { RunDetail } from "./components/RunDetail";
import { RunHeader } from "./components/RunHeader";
import { RunNowButton } from "./components/RunNowButton";
import { RunSetup } from "./components/RunSetup";
import { RunsList } from "./components/RunsList";
import { useWorkflowRun, type WorkflowTab } from "./useWorkflowRun";

interface Props {
  tab: WorkflowTab;
}

/** A workflow beside the chat, as two pages: its runs, newest first, with
 *  Run now and Schedule; and one run, opened from the list or as soon as
 *  it starts, with its input, trace and output. */
export function WorkflowRunTab({ tab }: Props) {
  const view = useWorkflowRun(tab);
  const [isSettingUp, setIsSettingUp] = useState(false);

  if (isSettingUp && view.agent) {
    return (
      <RunSetup
        agent={view.agent}
        onCancel={() => setIsSettingUp(false)}
        onStarted={(executionID) => {
          setIsSettingUp(false);
          view.showRun(executionID);
        }}
      />
    );
  }

  if (view.selectedID) {
    return (
      <RunDetail
        graphID={tab.graphID}
        title={view.title}
        runID={view.selectedID}
        run={view.run}
        trace={view.trace}
        isLive={view.isLive}
        now={view.now}
        onBack={view.showRuns}
      />
    );
  }

  return (
    <div aria-label={`Workflow: ${view.title}`} className="flex flex-col gap-5">
      <RunHeader
        graphID={tab.graphID}
        title={view.title}
        agent={view.agent}
        run={view.runs[0]}
        onStarted={view.showRun}
        onSetup={() => setIsSettingUp(true)}
      />

      <InfoGroup label="Runs" count={view.runs.length}>
        {view.isLoadingRuns ? (
          <InfoLoading />
        ) : view.runs.length === 0 ? (
          <BlockNote className="items-start gap-2.5">
            <Text variant="inbox" tone="quiet">
              This workflow has not run yet.
            </Text>
            <RunNowButton
              agent={view.agent}
              onStarted={view.showRun}
              onSetup={() => setIsSettingUp(true)}
            />
          </BlockNote>
        ) : (
          <InfoList>
            <RunsList runs={view.runs} now={view.now} onSelect={view.showRun} />
          </InfoList>
        )}
      </InfoGroup>
    </div>
  );
}
