"use client";

import {
  CheckmarkCircle02Icon,
  SparklesIcon,
} from "@hugeicons/core-free-icons";

import type { ChatTransportResponse } from "@/app/api/__generated__/models/chatTransportResponse";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";

import { describeTransport, transportKey } from "./helpers";
import { useAIConnectionsSection } from "./useAIConnectionsSection";

export function AIConnectionsSection() {
  const {
    connections,
    selectedKey,
    chooseDefault,
    isSaving,
    isLoading,
    isError,
  } = useAIConnectionsSection();

  // A failure here must not take the tool integrations below it down with it.
  if (isError) return null;

  // One connection is not a choice. The rows still say what powers a chat,
  // they just stop presenting a decision that doesn't exist.
  const hasChoice = connections.length > 1;

  return (
    <section aria-labelledby="ai-connections-heading" className="pb-8 pl-4">
      <Text
        variant="small-medium"
        as="h2"
        id="ai-connections-heading"
        className="uppercase tracking-[0.06em] text-[#505057]"
      >
        AI subscriptions
      </Text>
      <Text variant="body" className="mt-2 max-w-[600px] text-[#505057]">
        {hasChoice
          ? "These power your agents. Pick the one new chats should start on — you can still change it per conversation, and nothing switches on its own."
          : "These power your agents. Link a subscription and you can choose which one new chats start on."}
      </Text>

      {isLoading ? (
        <div className="mt-4 flex flex-col gap-3">
          <Skeleton className="h-[86px] w-full rounded-2xl" />
          <Skeleton className="h-[86px] w-full rounded-2xl" />
        </div>
      ) : (
        <div
          role={hasChoice ? "radiogroup" : undefined}
          aria-label={hasChoice ? "Connection new chats start on" : undefined}
          className="mt-4 flex flex-col gap-3"
        >
          {connections.map((connection) => (
            <ConnectionRow
              key={transportKey(connection)}
              connection={connection}
              selectable={hasChoice}
              isSelected={transportKey(connection) === selectedKey}
              isSaving={isSaving}
              onSelect={() => chooseDefault(connection)}
            />
          ))}
        </div>
      )}

      <UpcomingConnections />
    </section>
  );
}

interface RowProps {
  connection: ChatTransportResponse;
  selectable: boolean;
  isSelected: boolean;
  isSaving: boolean;
  onSelect: () => void;
}

function ConnectionRow({
  connection,
  selectable,
  isSelected,
  isSaving,
  onSelect,
}: RowProps) {
  const body = (
    <>
      {selectable && (
        <span
          aria-hidden
          className={cn(
            "mt-[3px] flex h-4 w-4 flex-none items-center justify-center rounded-full border",
            isSelected ? "border-[#7444E5]" : "border-[#9A9A9F]",
          )}
        >
          {isSelected && (
            <span className="h-2 w-2 rounded-full bg-[#7444E5]" aria-hidden />
          )}
        </span>
      )}

      <span className="flex min-w-0 flex-col gap-1">
        <span className="flex flex-wrap items-center gap-2">
          <Text variant="body-medium" as="span" className="text-black">
            {connection.label}
          </Text>
          {connection.auth_provider === "codex" && (
            <span className="inline-flex items-center gap-1 rounded-[10px] bg-[#E8F8F0] px-2 py-[2px] text-[13px] font-medium leading-[20px] text-[#157E58]">
              <Icon icon={CheckmarkCircle02Icon} size={13} />
              Connected
            </span>
          )}
          {isSelected && (
            <span className="inline-flex items-center gap-1 rounded-[10px] bg-[#F1EBFF] px-2 py-[2px] text-[13px] font-medium leading-[20px] text-[#4A25AD]">
              <Icon icon={SparklesIcon} size={13} />
              Used for new chats
            </span>
          )}
        </span>
        <Text variant="small" as="span" className="text-[#505057]">
          {describeTransport(connection)}
        </Text>
      </span>
    </>
  );

  if (!selectable) {
    return (
      <div className="flex w-full items-start gap-3 rounded-2xl border border-[#DADADC] bg-white p-4">
        {body}
      </div>
    );
  }

  return (
    <button
      type="button"
      role="radio"
      aria-checked={isSelected}
      disabled={isSaving}
      onClick={onSelect}
      className={cn(
        "flex w-full items-start gap-3 rounded-2xl border bg-white p-4 text-left transition-colors",
        "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#7444E5] focus-visible:ring-offset-2",
        isSelected
          ? "border-[#7444E5] ring-1 ring-[#7444E5]"
          : "border-[#DADADC] hover:bg-[#F9F9FA]",
        isSaving && "cursor-progress opacity-70",
      )}
    >
      {body}
    </button>
  );
}

/**
 * Names what is coming without claiming it works yet. Each provider needs its
 * own adapter and its own provider/legal approval before it can appear as a
 * real row above, so this promises nothing about capability or timing.
 */
function UpcomingConnections() {
  return (
    <div className="mt-3 flex items-start gap-3 rounded-2xl border border-dashed border-[#DADADC] p-4">
      <span
        aria-hidden
        className="mt-[2px] flex h-4 w-4 flex-none items-center justify-center text-[#9A9A9F]"
      >
        <Icon icon={SparklesIcon} size={16} />
      </span>
      <span className="flex min-w-0 flex-col gap-1">
        <span className="flex flex-wrap items-center gap-2">
          <Text variant="body-medium" as="span" className="text-[#505057]">
            GitHub Copilot and Grok
          </Text>
          <span className="inline-flex items-center rounded-[10px] bg-[#EFF1F4] px-2 py-[2px] text-[13px] font-medium leading-[20px] text-[#505057]">
            Coming soon
          </span>
        </span>
        <Text variant="small" as="span" className="text-[#505057]">
          More subscriptions you already pay for. Each one shows up here once it
          is approved to run AutoGPT agents.
        </Text>
      </span>
    </div>
  );
}
