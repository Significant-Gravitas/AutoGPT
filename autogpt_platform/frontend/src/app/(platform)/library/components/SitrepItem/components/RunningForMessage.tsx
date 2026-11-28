"use client";

import { runningMessage } from "@/app/(platform)/library/hooks/executionHelpers";
import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import { OverflowText } from "@/components/atoms/OverflowText/OverflowText";
import { useNowSecond } from "@/hooks/useNowSecond";

interface Props {
  startedAt: Date | string;
}

export function RunningForMessage({ startedAt }: Props) {
  const nowMs = useNowSecond();

  return (
    <OverflowText
      value={runningMessage(AgentExecutionStatus.RUNNING, startedAt, nowMs)}
      variant="small"
      leading="tight"
      tone="muted"
    />
  );
}
