"use client";

import type { Trace } from "@/app/(platform)/copilot/components/WorkArea/WorkflowRunTab/trace";
import type { GraphExecution } from "@/app/api/__generated__/models/graphExecution";

import {
  AgentRunDetailScreen,
  StepView,
  type RunStep,
} from "../affect/agent-run-detail";
import { toAgentRun, withoutError } from "../helpers";
import { RunIO } from "./RunIO";
import { RunValues } from "./RunValues";

export interface RunScreenData {
  graphID: string;
  title: string;
  run: GraphExecution | undefined;
  trace: Trace;
  isLive: boolean;
  now: number;
}

interface Props extends RunScreenData {
  opened: string | null;
  onOpen: (stepID: string) => void;
  onRerun: () => boolean;
  onStop: () => void;
  /** In the chat's sidebar, which scrolls it. */
  embed?: boolean;
}

/** One workflow run on the Affect UI Run Detail block: its head, four
 *  figures and the block runs on one time axis, then what it was given and
 *  what it gave back. */
export function RunScreen({
  opened,
  onOpen,
  onRerun,
  onStop,
  embed = false,
  ...data
}: Props) {
  if (!data.run) {
    return <AgentRunDetailScreen loading embed={embed} />;
  }
  const { agentRun, elapsed } = toAgentRun({ ...data, run: data.run });
  return (
    <AgentRunDetailScreen
      run={agentRun}
      elapsed={elapsed}
      opened={opened}
      onOpen={onOpen}
      onRerun={onRerun}
      onStop={onStop}
      empty={agentRun.steps.length === 0}
      embed={embed}
    >
      <RunIO
        inputs={data.run.inputs ?? null}
        outputs={data.isLive ? undefined : (data.run.outputs ?? null)}
      />
    </AgentRunDetailScreen>
  );
}

interface StepProps extends RunScreenData {
  stepID: string | null;
  embed?: boolean;
}

/** One block run in the block's step panel, its values drawn to be read. */
export function RunStepView({ stepID, embed = false, ...data }: StepProps) {
  if (!data.run) return null;
  const { agentRun, elapsed } = toAgentRun({ ...data, run: data.run });
  const traced = data.trace.steps.find((step) => step.id === stepID);
  const step = agentRun.steps.find((entry) => entry.id === stepID);
  const shown: RunStep | null =
    step && traced
      ? {
          ...step,
          inputView: <RunValues value={traced.input} />,
          outputView: <RunValues value={withoutError(traced.output)} />,
        }
      : null;
  return (
    <StepView run={agentRun} step={shown} elapsed={elapsed} embed={embed} />
  );
}
