import React from "react";
import { Text } from "@/components/atoms/Text/Text";
import { Refresh01Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

interface LoadingStateProps {
  loadingSlot?: React.ReactNode;
}

export function LoadingState({ loadingSlot }: LoadingStateProps) {
  return (
    <div className="relative flex items-center justify-center gap-3 p-6">
      {loadingSlot || (
        <>
          <Icon
            icon={Refresh01Icon}
            size={20}
            className="animate-spin text-purple-500"
          />
          <Text variant="body" className="text-zinc-600">
            Loading...
          </Text>
        </>
      )}
    </div>
  );
}
