"use client";

import { useState } from "react";
import {
  CheckmarkCircle02Icon,
  SparklesIcon,
} from "@hugeicons/core-free-icons";

import type { AIConnectionOffer } from "@/app/api/__generated__/models/aIConnectionOffer";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";

import { ManageConnectionDialog } from "./ManageConnectionDialog";
import { isSelectable, tierSummary } from "./helpers";
import { useAIConnectionsSection } from "./useAIConnectionsSection";

export function AIConnectionsSection() {
  const {
    connections,
    accountFor,
    selectedKey,
    chooseDefault,
    isSaving,
    isLoading,
    isError,
  } = useAIConnectionsSection();
  const [managing, setManaging] = useState<AIConnectionOffer | null>(null);

  // A failure here must not take the tool integrations below it down with it.
  if (isError) return null;

  // One connection is not a choice. The rows still say what powers a chat,
  // they just stop presenting a decision that doesn't exist.
  //
  // Counted over what can actually be picked: a locked upsell row is listed
  // so the user can see the connection exists, but offering it as a choice
  // produces a click the server can only refuse.
  const selectableCount = connections.filter(isSelectable).length;
  const hasChoice = selectableCount > 1;

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
              key={connection.offer_id}
              connection={connection}
              account={accountFor(connection)}
              selectable={hasChoice && isSelectable(connection)}
              isSelected={connection.offer_id === selectedKey}
              isSaving={isSaving}
              onSelect={() => chooseDefault(connection)}
              onManage={
                connection.credential_id &&
                connection.auth_method === "chatgpt_oauth"
                  ? () => setManaging(connection)
                  : undefined
              }
            />
          ))}
        </div>
      )}

      <UpcomingConnections />

      <ManageConnectionDialog
        connection={managing}
        account={managing ? accountFor(managing) : undefined}
        onOpenChange={(open) => {
          if (!open) setManaging(null);
        }}
      />
    </section>
  );
}

interface RowProps {
  connection: AIConnectionOffer;
  account?: string;
  selectable: boolean;
  isSelected: boolean;
  isSaving: boolean;
  onSelect: () => void;
  onManage?: () => void;
}

function ConnectionRow({
  connection,
  account,
  selectable,
  isSelected,
  isSaving,
  onSelect,
  onManage,
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
            {connection.display_name}
          </Text>
          {connection.auth_method === "chatgpt_oauth" &&
            isSelectable(connection) && (
              <span className="inline-flex items-center gap-1 rounded-[10px] bg-[#E8F8F0] px-2 py-[2px] text-[13px] font-medium leading-[20px] text-[#157E58]">
                <Icon icon={CheckmarkCircle02Icon} size={13} />
                Connected
              </span>
            )}
          {account && (
            <span className="max-w-full truncate rounded-[10px] bg-[#EFF1F4] px-2 py-[2px] text-[13px] font-medium leading-[20px] text-[#505057]">
              {account}
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
          {connection.description}
        </Text>
        {tierSummary(connection) && (
          <Text variant="small" as="span" className="text-[#7A7A80]">
            {tierSummary(connection)}
          </Text>
        )}
        {connection.lock_reason && (
          <Text variant="small" as="span" className="text-[#7A7A80]">
            {connection.lock_reason}
          </Text>
        )}
      </span>
    </>
  );

  const manage = onManage ? (
    <Button
      variant="secondary"
      size="small"
      className="ml-auto flex-none self-center"
      onClick={onManage}
    >
      Manage
    </Button>
  ) : null;

  if (!selectable) {
    return (
      <div className="flex w-full items-start gap-3 rounded-2xl border border-[#DADADC] bg-white p-4">
        {body}
        {manage}
      </div>
    );
  }

  return (
    <div
      className={cn(
        "flex w-full items-start rounded-2xl border bg-white pr-4 transition-colors",
        isSelected
          ? "border-[#7444E5] ring-1 ring-[#7444E5]"
          : "border-[#DADADC] hover:bg-[#F9F9FA]",
      )}
    >
      <button
        type="button"
        role="radio"
        aria-checked={isSelected}
        disabled={isSaving}
        onClick={onSelect}
        className={cn(
          "flex min-w-0 flex-1 items-start gap-3 rounded-2xl p-4 text-left",
          "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#7444E5]",
          isSaving && "cursor-progress opacity-70",
        )}
      >
        {body}
      </button>
      {manage}
    </div>
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
