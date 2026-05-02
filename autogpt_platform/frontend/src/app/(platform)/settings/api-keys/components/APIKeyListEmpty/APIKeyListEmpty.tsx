"use client";

import { Text } from "@/components/atoms/Text/Text";

import { APIKeyMarquee } from "./components/APIKeyMarquee";

export function APIKeyListEmpty() {
  return (
    <div className="flex flex-col items-center justify-center gap-4 px-6 py-10 text-center">
      <APIKeyMarquee />
      <div className="flex flex-col items-center gap-1">
        <Text variant="large-medium" as="span" className="text-textBlack">
          No API key found
        </Text>
        <Text variant="body" className="max-w-[360px] text-zinc-500">
          You haven&apos;t created an API key yet. Create one to start using the
          AutoGPT Platform API.
        </Text>
      </div>
    </div>
  );
}
