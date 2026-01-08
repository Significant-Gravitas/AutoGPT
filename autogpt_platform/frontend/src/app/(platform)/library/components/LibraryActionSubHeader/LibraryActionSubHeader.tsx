"use client";

import { Text } from "@/components/atoms/Text/Text";
import { LibrarySortMenu } from "../LibrarySortMenu/LibrarySortMenu";

interface Props {
  agentCount: number;
}

export function LibraryActionSubHeader({ agentCount }: Props) {
  return (
    <div className="flex items-baseline justify-between">
      <div className="flex items-baseline gap-4">
        <Text variant="h4">My agents</Text>
        <Text
          variant="body"
          data-testid="agents-count"
          className="text-zinc-500"
        >
          {agentCount}
        </Text>
      </div>
      <LibrarySortMenu />
    </div>
  );
}
