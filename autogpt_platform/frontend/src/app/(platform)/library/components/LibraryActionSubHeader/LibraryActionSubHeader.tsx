"use client";

import { LibraryAgentSort } from "@/app/api/__generated__/models/libraryAgentSort";
import { Text } from "@/components/atoms/Text/Text";
import { LibrarySortMenu } from "../LibrarySortMenu/LibrarySortMenu";

interface Props {
  agentCount: number;
  setLibrarySort: (value: LibraryAgentSort) => void;
}

export function LibraryActionSubHeader({ agentCount, setLibrarySort }: Props) {
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
      <LibrarySortMenu setLibrarySort={setLibrarySort} />
    </div>
  );
}
