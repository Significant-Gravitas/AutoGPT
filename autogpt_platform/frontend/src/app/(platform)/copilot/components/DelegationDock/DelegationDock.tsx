"use client";

import {
  Alert02Icon,
  ArrowRight01Icon,
  CheckmarkCircle02Icon,
  Loading03Icon,
  MessageQuestionIcon,
} from "@hugeicons/core-free-icons";
import type { IconSvgElement } from "@hugeicons/react";
import type { UIDataTypes, UIMessage, UITools } from "ai";
import { useState } from "react";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import {
  type LiveDelegationStatus,
  getChatDelegations,
} from "../../delegations";
import { useCopilotUIStore } from "../../store";
import { LiveDelegationProbes } from "../DelegationStatusLine/LiveDelegationProbes";
import { getDockLine, type DockLine } from "./helpers";

const DOCK_ICON: Record<
  DockLine["tone"],
  { icon: IconSvgElement; className: string }
> = {
  working: { icon: Loading03Icon, className: "animate-spin text-purple-500" },
  waiting: { icon: MessageQuestionIcon, className: "text-amber-500" },
  done: { icon: CheckmarkCircle02Icon, className: "text-emerald-600" },
  failed: { icon: Alert02Icon, className: "text-red-500" },
};

interface Props {
  messages: UIMessage<unknown, UIDataTypes, UITools>[];
  /** The task progress bar holds this slot while a task list runs. */
  hasActiveTaskList?: boolean;
}

/** The one-line bar docked above the composer while experts are at work:
 *  the same shell as the task progress bar, one sentence about the experts,
 *  and a click that opens the Work panel. */
export function DelegationDock({ messages, hasActiveTaskList = false }: Props) {
  const openWorkTab = useCopilotUIStore((s) => s.openWorkTab);
  const [statuses, setStatuses] = useState<
    Record<string, LiveDelegationStatus>
  >({});
  function handleStatus(toolCallId: string, status: LiveDelegationStatus) {
    setStatuses((prev) =>
      prev[toolCallId] === status ? prev : { ...prev, [toolCallId]: status },
    );
  }
  if (hasActiveTaskList) return null;
  const delegations = getChatDelegations(messages);
  const line = getDockLine(delegations, statuses);

  return (
    <>
      <LiveDelegationProbes delegations={delegations} onStatus={handleStatus} />
      {line && (
        <div className="mx-auto w-[95%] overflow-hidden rounded-t-3xl border border-b-0 border-zinc-200 bg-neutral-100 shadow-[inset_0_1px_0_0_rgba(255,255,255,0.9),inset_0_5px_6px_-4px_rgba(255,255,255,0.7)]">
          <button
            type="button"
            data-testid="delegation-dock"
            onClick={openWorkTab}
            className="flex w-full items-center gap-2 px-3 py-3 text-left"
          >
            <Icon
              icon={DOCK_ICON[line.tone].icon}
              size={16}
              className={cn(
                "flex-shrink-0 motion-reduce:animate-none",
                DOCK_ICON[line.tone].className,
              )}
            />
            <Text
              variant="body-medium"
              className="min-w-0 flex-1 truncate text-sm text-zinc-800"
            >
              {line.text}
            </Text>
            <span className="flex flex-shrink-0 items-center gap-1 text-xs text-zinc-600">
              Open work
              <Icon
                icon={ArrowRight01Icon}
                size={14}
                className="text-zinc-500"
              />
            </span>
          </button>
        </div>
      )}
    </>
  );
}
