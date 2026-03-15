import React from "react";
import { Text } from "@/components/atoms/Text/Text";
import { Warning } from "@phosphor-icons/react";

export function ErrorHeader() {
  return (
    <div className="flex items-center gap-3">
      <div className="flex-shrink-0">
        <Warning size={24} weight="fill" className="text-red-400" />
      </div>
      <div>
        <Text variant="large-semibold" className="text-zinc-800">
          Something went wrong
        </Text>
      </div>
    </div>
  );
}
