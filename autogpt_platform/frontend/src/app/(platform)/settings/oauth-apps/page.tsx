"use client";

import { Text } from "@/components/atoms/Text/Text";

export default function SettingsOAuthAppsPage() {
  return (
    <>
      <Text
        // TODO(tokens): no exact role (text-[#1F1F20])
        variant="h4"
        className="text-[#1F1F20]"
      >
        OAuth Apps
      </Text>
      <Text variant="large" className="mt-2 text-foreground-secondary">
        Coming soon.
      </Text>
    </>
  );
}
