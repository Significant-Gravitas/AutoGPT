"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { LockIcon } from "@hugeicons/core-free-icons";

const PRIVACY_COPY =
  "Private to you · Saved as Otto memory · Downloadable anytime";

export function PrivacyNote() {
  return (
    <div className="flex items-center justify-center gap-2 px-4 text-foreground-subtle">
      <Icon icon={LockIcon} size={14} className="shrink-0" />
      <Text
        variant="small"
        tone="subtle"
        className="text-center sm:whitespace-nowrap"
      >
        {PRIVACY_COPY}
      </Text>
    </div>
  );
}
