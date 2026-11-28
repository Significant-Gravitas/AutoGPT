"use client";

import { Text } from "@/components/atoms/Text/Text";

export function OAuthAppsHeader() {
  return (
    <div className="flex flex-col pb-6 pl-4">
      <Text variant="h4" as="h1">
        OAuth Apps
      </Text>
      <Text variant="body" tone="body" className="mt-4 max-w-[600px]">
        Apps you&apos;ve allowed to sign in with your AutoGPT account. Revoking
        an app ends its access right away.
      </Text>
    </div>
  );
}
