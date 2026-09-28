"use client";

import {
  Alert02Icon,
  ArrowReloadHorizontalIcon,
  ArrowRight01Icon,
  CheckmarkCircle02Icon,
  CircleIcon,
  Loading03Icon,
  MessageQuestionIcon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";
import type { UIMessage } from "ai";
import { useContext } from "react";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { cn } from "@/lib/utils";
import type { MessagePart } from "../ChatMessagesContainer/helpers";
import { CopilotChatActionsContext } from "../CopilotChatActionsProvider/useCopilotChatActions";
import {
  type ChatDelegation,
  type DelegationTone,
  getChatDelegations,
  getDelegationStatusView,
} from "../../delegations";
import { useCopilotUIStore } from "../../store";
import { useDelegationLive } from "../../useDelegationLive";
import { getStatusLineText, retryMessage } from "./helpers";

const TONE_ICON: Record<
  DelegationTone,
  { icon: IconSvgElement; className: string }
> = {
  working: { icon: Loading03Icon, className: "animate-spin text-purple-500" },
  waiting: { icon: MessageQuestionIcon, className: "text-amber-500" },
  done: { icon: CheckmarkCircle02Icon, className: "text-emerald-600" },
  failed: { icon: Alert02Icon, className: "text-red-500" },
  muted: { icon: CircleIcon, className: "text-zinc-400" },
};

interface LineProps {
  delegation: ChatDelegation;
  readOnly: boolean;
}

function DelegationLine({ delegation, readOnly }: LineProps) {
  const live = useDelegationLive(delegation);
  const openWorkTab = useCopilotUIStore((s) => s.openWorkTab);
  const actions = useContext(CopilotChatActionsContext);
  const view = getDelegationStatusView(live.status);
  const text = getStatusLineText(
    delegation,
    live.status,
    live.elapsedSeconds,
    live.question,
  );
  const tone = TONE_ICON[view.tone];
  const canRetry = live.status === "failed" && !readOnly && !!actions;

  return (
    <div
      data-testid="delegation-status-line"
      data-status={live.status}
      className="flex h-10 items-center gap-2.5 rounded-lg pl-1 pr-2"
    >
      <Icon
        icon={tone.icon}
        size={16}
        className={cn("shrink-0 motion-reduce:animate-none", tone.className)}
      />
      <span className="shrink-0 text-sm font-medium text-zinc-800">
        {text.headline}
      </span>
      {text.detail && (
        <span className="min-w-0 truncate text-sm text-zinc-500">
          · {text.detail}
        </span>
      )}
      <span className="flex-1" />
      {canRetry && (
        <Button
          variant="secondary"
          size="xs"
          leadingIcon={ArrowReloadHorizontalIcon}
          onClick={() => void actions.onSend(retryMessage(delegation))}
        >
          Retry
        </Button>
      )}
      <button
        type="button"
        onClick={openWorkTab}
        className="flex shrink-0 items-center gap-1 rounded-md px-1 text-xs text-zinc-600 transition-colors hover:text-zinc-900"
      >
        Open
        <Icon icon={ArrowRight01Icon} size={14} className="text-zinc-500" />
      </button>
    </div>
  );
}

interface Props {
  parts: MessagePart[];
  messageId: string;
  readOnly?: boolean;
}

/** One line per hand-off under a turn's chain: a loader, who is on it and
 *  for how long, and the way into the Work panel. It replaces the big
 *  delegation card in the thread once the hand-off is under way. */
export function DelegationStatusLine({
  parts,
  messageId,
  readOnly = false,
}: Props) {
  const delegations = getChatDelegations([
    { id: messageId, role: "assistant", parts } as UIMessage,
  ]).filter((delegation) => delegation.status !== "proposed");
  if (delegations.length === 0) return null;
  return (
    <div className="my-1 flex flex-col">
      {delegations.map((delegation) => (
        <DelegationLine
          key={delegation.toolCallId}
          delegation={delegation}
          readOnly={readOnly}
        />
      ))}
    </div>
  );
}
