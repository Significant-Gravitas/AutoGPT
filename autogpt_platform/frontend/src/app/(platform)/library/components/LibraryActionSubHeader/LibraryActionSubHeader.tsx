"use client";

import { Text } from "@/components/atoms/Text/Text";

interface Props {
  agentCount: number;
}

export function LibraryActionSubHeader({ agentCount }: Props) {
  return (
    <div className="flex items-baseline justify-between">
      <Text variant="body" data-testid="agents-count" className="text-zinc-500">
        {agentCount}
      </Text>
    </div>
  );
}
