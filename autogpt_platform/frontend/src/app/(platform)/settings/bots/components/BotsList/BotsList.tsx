"use client";

import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Text } from "@/components/atoms/Text/Text";

import { BotCard } from "../BotCard/BotCard";
import { useBotsList } from "./useBotsList";

export function BotsList() {
  const { platforms, isLoading, isError, error, refetch, isEmpty } =
    useBotsList();

  if (isLoading) {
    return (
      <div className="flex w-full flex-col gap-3 px-4">
        <Skeleton className="h-40 rounded-xl" />
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
        <Text variant="large-medium" as="span">
          No bots enabled
        </Text>
        <Text variant="body" className="max-w-[360px] text-zinc-500">
          No chat-bot platforms are available on this deployment right now.
        </Text>
      </div>
    );
  }

  return (
    <div className="grid w-full grid-cols-1 items-start gap-4 px-4 pb-4 lg:grid-cols-2">
      {platforms.map((platform) => (
        <BotCard key={platform.platform} platform={platform} />
      ))}
    </div>
  );
}
