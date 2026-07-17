"use client";

import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { Text } from "@/components/atoms/Text/Text";
import { RunDetailCard } from "../selected-views/RunDetailCard/RunDetailCard";
import { SelectedViewLayout } from "../selected-views/SelectedViewLayout";

interface Props {
  agent: LibraryAgent;
  banner?: React.ReactNode;
}

export function TriggerNotFound({ agent, banner }: Props) {
  return (
    <SelectedViewLayout agent={agent} banner={banner}>
      <RunDetailCard title="Trigger not found">
        <Text variant="body" className="!text-zinc-500">
          This trigger no longer exists. It may have been removed or replaced
          when the agent was updated.
        </Text>
      </RunDetailCard>
    </SelectedViewLayout>
  );
}
