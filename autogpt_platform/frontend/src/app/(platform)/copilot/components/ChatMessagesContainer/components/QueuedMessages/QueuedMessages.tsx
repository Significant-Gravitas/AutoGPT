import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Message, MessageContent } from "@/components/ai-elements/message";
import { cn } from "@/lib/utils";
import {
  Alert02Icon,
  Clock01Icon,
  RefreshIcon,
} from "@hugeicons/core-free-icons";

import type { HeldReason } from "../../../../useCopilotPendingChips";
import { CredentialMentionText } from "../../../CredentialMention/CredentialMentionText";

interface Props {
  messages: string[];
  /** Set when nothing will answer these messages unless they are resent. */
  heldReason: HeldReason | null;
  onRetry?: (text: string) => void;
  isCompact: boolean;
}

export function QueuedMessages({
  messages,
  heldReason,
  onRetry,
  isCompact,
}: Props) {
  const isHeld = heldReason !== null && onRetry !== undefined;
  return messages.map((msg, idx) => (
    <Message key={idx} from="user">
      <MessageContent
        className={cn(
          "flex flex-col gap-1 border px-4 py-2.5 text-zinc-900",
          isHeld
            ? "border-red-300 bg-white"
            : "border-dashed border-zinc-300 bg-zinc-100 opacity-60",
          isCompact
            ? "rounded-xl text-sm leading-6"
            : "rounded-3xl text-[1rem] leading-relaxed",
        )}
      >
        <span>
          <CredentialMentionText text={msg} />
        </span>
        {isHeld ? (
          <span className="flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-red-700">
            <Icon icon={Alert02Icon} className="size-3" />
            <span>Didn&apos;t go through</span>
            <span className="text-zinc-500">{HELD_HINTS[heldReason]}</span>
            <Button
              variant="secondary"
              size="xs"
              leftIcon={<Icon icon={RefreshIcon} />}
              onClick={() => onRetry(msg)}
            >
              Retry
            </Button>
          </span>
        ) : (
          <span className="flex items-center gap-1 text-xs text-slate-500">
            <Icon icon={Clock01Icon} className="size-3" />
            <span>Queued</span>
            <span>· Otto will answer after the current task</span>
          </span>
        )}
      </MessageContent>
    </Message>
  ));
}

const HELD_HINTS: Record<HeldReason, string> = {
  error: "Otto hit a problem before reading it.",
  stopped: "Otto was stopped before reading it.",
  paused: "Otto paused before reading it.",
  ended: "Otto finished before reading it.",
};
