"use client";

import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { LockIcon } from "@hugeicons/core-free-icons";

const PRIVACY_COPY =
  "Private to you · Saved as AutoPilot memory · Downloadable anytime";

export function PrivacyNote() {
  return (
    <div className="fixed inset-x-0 bottom-20 flex items-center justify-center gap-2 px-4">
      <Icon icon={LockIcon} size={14} className="shrink-0 text-zinc-400" />
      <Text
        variant="small"
        className="text-center !text-sm !text-zinc-400 sm:whitespace-nowrap"
      >
        {PRIVACY_COPY}
      </Text>
    </div>
  );
}
