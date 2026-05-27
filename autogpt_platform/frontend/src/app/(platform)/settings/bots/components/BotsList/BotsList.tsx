"use client";

import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { Text } from "@/components/atoms/Text/Text";

import { BotCard } from "../BotCard/BotCard";
import { useBotsList } from "./useBotsList";

export function BotsList() {
  const { platforms, isLoading, isError, error, refetch, isEmpty } =
    useBotsList();

  if (isLoading) {
    return (
      <div className="flex w-full flex-col gap-3 px-4">
        <div className="h-40 animate-pulse rounded-large bg-zinc-100" />
      </div>
    );
  }

  if (isError) {
    return (
      <div className="flex w-full flex-col gap-3 px-4">
        <ErrorCard
          context="bots"
          responseError={
            error instanceof Error ? { message: error.message } : undefined
          }
          onRetry={() => refetch()}
        />
      </div>
    );
  }

  if (isEmpty) {
    return (
      <div className="flex flex-col items-center justify-center gap-2 px-6 py-10 text-center">
        <Text variant="large-medium" as="span" className="text-textBlack">
          No bots enabled
        </Text>
        <Text variant="body" className="max-w-[360px] text-zinc-500">
          No chat-bot platforms are configured on this deployment. Ask an
          administrator to enable Discord, Slack, or another adapter.
        </Text>
      </div>
    );
  }

  return (
    <div className="flex w-full flex-col gap-4 px-4 pb-4">
      {platforms.map((platform) => (
        <BotCard key={platform.platform} platform={platform} />
      ))}
    </div>
  );
}
