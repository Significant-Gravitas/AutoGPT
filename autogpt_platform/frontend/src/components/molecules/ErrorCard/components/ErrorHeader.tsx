import React from "react";
import { Text } from "@/components/atoms/Text/Text";
import { Alert01Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

export function ErrorHeader() {
  return (
    <div data-slot="error-card-header" className="flex items-center gap-3">
      <div data-slot="error-card-icon-wrap" className="flex-shrink-0">
        {/* TODO(tokens): no role for red-400 (#F26969). */}
        <Icon
          data-slot="error-card-icon"
          icon={Alert01Icon}
          size={24}
          className="text-red-400"
        />
      </div>
      <div data-slot="error-card-heading">
        <Text
          variant="large-semibold"
          data-slot="error-card-title"
          className="text-foreground-emphasis"
        >
          Something went wrong
        </Text>
      </div>
    </div>
  );
}
