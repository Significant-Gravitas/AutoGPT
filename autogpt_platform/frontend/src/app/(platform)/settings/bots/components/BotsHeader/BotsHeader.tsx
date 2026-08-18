"use client";

import { Text } from "@/components/atoms/Text/Text";

export function BotsHeader() {
  return (
    <div className="flex flex-col items-start gap-4 pb-6 pl-4 sm:flex-row sm:items-start sm:justify-between">
      <div className="flex min-w-0 flex-col">
        <Text variant="h4" as="h1" className="leading-[28px] text-zinc-900">
          Bots
        </Text>
        <Text variant="body" className="mt-4 max-w-[600px] text-zinc-600">
          Connect AutoGPT bots to your chat platforms. Add a bot to one of your
          servers, link a DM channel so the bot can talk to you directly, or
          unlink an existing connection.
        </Text>
      </div>
    </div>
  );
}
