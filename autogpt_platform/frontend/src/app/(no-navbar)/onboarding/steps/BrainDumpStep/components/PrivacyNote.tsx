"use client";

import { Text } from "@/components/atoms/Text/Text";
import { LockSimpleIcon } from "@phosphor-icons/react";

const PRIVACY_COPY =
  "Private to you · Saved as AutoPilot memory · Downloadable anytime";

export function PrivacyNote() {
  return (
    <div className="fixed inset-x-0 bottom-20 flex items-center justify-center gap-2 px-4">
      <LockSimpleIcon size={14} className="shrink-0 text-zinc-400" />
      <Text
        variant="small"
        className="text-center !text-sm !text-zinc-400 sm:whitespace-nowrap"
      >
        {PRIVACY_COPY}
      </Text>
    </div>
  );
}
