"use client";

import { WorkflowPage } from "@/app/(platform)/library/agents/[id]/components/NewAgentLibraryView/components/WorkflowPage/WorkflowPage";
import { Button } from "@/components/atoms/Button/Button";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import type { ToolUIPart } from "ai";
import NextLink from "next/link";
import { useState } from "react";
import { useCopilotChatActions } from "../../components/CopilotChatActionsProvider/useCopilotChatActions";
import { GenericTool } from "../GenericTool/GenericTool";
import { APPLY_MESSAGE, DISCARD_MESSAGE, parseWorkflowPage } from "./helpers";

interface Props {
  part: ToolUIPart;
  canAct: boolean;
  isCurrentlyStreaming?: boolean;
}

export function DesignWorkflowPage({
  part,
  canAct,
  isCurrentlyStreaming,
}: Props) {
  const { onSend } = useCopilotChatActions();
  const [isSent, setIsSent] = useState(false);
  const output = parseWorkflowPage(part);

  if (!output) {
    const isPending =
      part.state === "input-streaming" || part.state === "input-available";
    if (isPending && isCurrentlyStreaming) {
      return <Skeleton className="my-2 h-48 w-full rounded-2xl" />;
    }
    return <GenericTool part={part} />;
  }

  function send(message: string) {
    if (!canAct || isSent) return;
    setIsSent(true);
    void onSend(message);
  }

  const libraryHref = `/library/agents/${output.page.library_agent_id}`;
  return (
    <div
      className="my-3 flex w-full min-w-0 flex-col gap-3 rounded-2xl bg-surface-muted p-4"
      data-testid="workflow-page-card"
    >
      <Text
        variant="small-medium"
        tone="secondary"
        transform="uppercase"
        tracking="caps"
      >
        {output.saved ? "Workflow page saved" : "Workflow page preview"}
      </Text>
      <WorkflowPage page={output.page} />
      {output.saved ? (
        <NextLink
          href={libraryHref}
          className="text-body font-medium underline"
        >
          Open it in the library
        </NextLink>
      ) : (
        <div className="flex gap-2">
          <Button
            size="small"
            disabled={!canAct || isSent}
            onClick={() => send(APPLY_MESSAGE)}
          >
            Apply
          </Button>
          <Button
            size="small"
            variant="outline"
            disabled={!canAct || isSent}
            onClick={() => send(DISCARD_MESSAGE)}
          >
            Discard
          </Button>
        </div>
      )}
    </div>
  );
}
