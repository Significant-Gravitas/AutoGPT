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

import { IntegrationLogo } from "@/components/molecules/IntegrationLogo/IntegrationLogo";
import { UpcomingProviderBoxes } from "./UpcomingProviderBoxes";
import { ProviderBox } from "./ProviderBox";
import { MicrosoftCopilotProviderBox } from "./MicrosoftCopilotProviderBox";
import { ManageConnectionDialog } from "./ManageConnectionDialog";
import { isSelectable, tierSummary } from "./helpers";
import { useAIConnectionsSection } from "./useAIConnectionsSection";

export function AIConnectionsSection() {
  const {
    connectChatGPT,
    isConnectingChatGPT,
    isChatGPTLinked,
    isMicrosoftLinked,
    connections,
    accountFor,
    credentialFor,
    selectedKey,
    chooseDefault,
    isSaving,
    isLoading,
    isError,
    refetch,
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
        tone="secondary"
        className="tracking-[0.06em] uppercase"
      >
        AI subscriptions
      </Text>
      <Text variant="body" tone="secondary" className="mt-2 max-w-[600px]">
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
                (connection.auth_provider === "codex" ||
                  connection.auth_provider === "microsoft_365_copilot")
                  ? () => setManaging(connection)
                  : undefined
              }
            />
          ))}
        </div>
      )}

      {!isLoading && (
        <div
          role="group"
          aria-label="Available AI subscriptions"
          className="mt-4 grid w-full grid-cols-2 gap-3 sm:grid-cols-4"
        >
          {!isChatGPTLinked && (
            <ProviderBox
              name="ChatGPT"
              logoSrc="/integrations/openai.png"
              state="available"
              isBusy={isConnectingChatGPT}
              onClick={connectChatGPT}
            />
          )}
          {!isMicrosoftLinked && (
            <MicrosoftCopilotProviderBox isLinked={false} onSuccess={refetch} />
          )}
          <UpcomingProviderBoxes />
        </div>
      )}

      <ManageConnectionDialog
        connection={managing}
        credential={managing ? credentialFor(managing) : undefined}
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
            isSelected ? "border-purple-500" : "border-zinc-400",
          )}
        >
          {isSelected && (
            <span className="h-2 w-2 rounded-full bg-purple-500" aria-hidden />
          )}
        </span>
      )}

      {(connection.auth_provider === "codex" ||
        connection.auth_provider === "microsoft_365_copilot") && (
        <IntegrationLogo
          provider={
            connection.auth_provider === "codex"
              ? "openai"
              : "microsoft_365_copilot"
          }
          alt=""
          size={32}
          className="shrink-0"
        />
      )}
      <span className="flex min-w-0 flex-col gap-1">
        <span className="flex flex-wrap items-center gap-2">
          <Text variant="body-medium" as="span">
            {connection.display_name}
          </Text>
          {connection.auth_method !== "deployment" &&
            connection.credential_id &&
            isSelectable(connection) && (
              <span className="inline-flex items-center gap-1 rounded-[10px] bg-green-50 px-2 py-0.5 text-[13px] leading-5 font-medium text-green-600">
                <Icon icon={CheckmarkCircle02Icon} size={13} />
                Connected
              </span>
            )}
          {account && (
            <span className="max-w-full truncate rounded-[10px] bg-slate-100 px-2 py-0.5 text-[13px] leading-5 font-medium text-zinc-700">
              {account}
            </span>
          )}
          {isSelected && (
            <span className="inline-flex items-center gap-1 rounded-[10px] bg-purple-50 px-2 py-0.5 text-[13px] leading-5 font-medium text-purple-800">
              <Icon icon={SparklesIcon} size={13} />
              Used for new chats
            </span>
          )}
        </span>
        <Text variant="small" as="span" tone="secondary">
          {connection.description}
        </Text>
        {tierSummary(connection) && (
          <Text variant="small" as="span" tone="muted">
            {tierSummary(connection)}
          </Text>
        )}
        {connection.lock_reason && (
          <Text variant="small" as="span" tone="muted">
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
      <div className="flex w-full items-start gap-3 rounded-2xl border border-zinc-200 bg-white p-4">
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
          ? "border-purple-500 ring-1 ring-purple-500"
          : "border-zinc-200 hover:bg-zinc-50",
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
          "focus-visible:ring-2 focus-visible:ring-purple-500 focus-visible:outline-hidden",
          isSaving && "cursor-progress opacity-70",
        )}
      >
        {body}
      </button>
      {manage}
    </div>
  );
}
