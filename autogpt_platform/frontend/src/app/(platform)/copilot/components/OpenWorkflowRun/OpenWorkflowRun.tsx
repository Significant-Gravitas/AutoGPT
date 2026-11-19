"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { ArrowRight01Icon, FlowIcon } from "@hugeicons/core-free-icons";
import { useContext } from "react";
import { useWorkAreaStore } from "../../workArea";
import { CopilotChatActionsContext } from "../CopilotChatActionsProvider/useCopilotChatActions";

interface Props {
  graphID: string | null | undefined;
  executionID?: string | null;
  title: string;
}

/** "See run" under a workflow's card in the chat: opens the run's trace in
 *  the sidebar. Only the signed-in chat has the sidebar, so the share page
 *  and the builder's panel leave it out. */
export function OpenWorkflowRun({ graphID, executionID, title }: Props) {
  const chatSurface = useContext(CopilotChatActionsContext)?.chatSurface;
  const openWorkflowRun = useWorkAreaStore((s) => s.openWorkflowRun);
  if (!graphID || (chatSurface && chatSurface !== "copilot")) return null;

  return (
    <button
      type="button"
      onClick={() =>
        openWorkflowRun({ graphID, executionID: executionID || null, title })
      }
      className="mt-1.5 inline-flex items-center gap-1.5 rounded-full bg-surface py-1 pr-2 pl-2.5 shadow-(--inbox-shadow-card) transition-[background-color] duration-150 ease-out-expo hover:bg-surface-wash focus-visible:ring-2 focus-visible:ring-focus-inbox focus-visible:outline-none"
    >
      <Icon icon={FlowIcon} size={13} tone="quiet" aria-hidden />
      <Text variant="inbox-medium" as="span" tone="ink">
        {executionID ? "See run" : "See runs"}
      </Text>
      <Icon icon={ArrowRight01Icon} size={12} tone="quiet" aria-hidden />
    </button>
  );
}
