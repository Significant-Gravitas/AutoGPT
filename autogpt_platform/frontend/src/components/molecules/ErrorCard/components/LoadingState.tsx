import React from "react";
import { Text } from "@/components/atoms/Text/Text";
import { ArrowClockwise } from "@phosphor-icons/react";

interface LoadingStateProps {
  loadingSlot?: React.ReactNode;
}

export function LoadingState({ loadingSlot }: LoadingStateProps) {
  return (
    <div className="relative flex items-center justify-center gap-3 p-6">
      {loadingSlot || (
        <>
          <ArrowClockwise
            size={20}
            weight="bold"
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
