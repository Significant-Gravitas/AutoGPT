"use client";

import type { LibraryAgent } from "@/app/api/__generated__/models/libraryAgent";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { RunDetailCard } from "../selected-views/RunDetailCard/RunDetailCard";
import { SelectedViewLayout } from "../selected-views/SelectedViewLayout";

interface Props {
  agent: LibraryAgent;
  banner?: React.ReactNode;
  onClearSelection?: () => void;
}

export function TriggerNotFound({ agent, banner, onClearSelection }: Props) {
  return (
    <SelectedViewLayout agent={agent} banner={banner}>
      <RunDetailCard title="Trigger not found">
        <div className="flex flex-col items-start gap-4">
          <Text variant="body" className="!text-zinc-500">
            This trigger doesn&apos;t exist or is no longer available.
          </Text>
          {onClearSelection && (
            <Button variant="secondary" size="small" onClick={onClearSelection}>
              Clear selection
            </Button>
          )}
        </div>
      </RunDetailCard>
    </SelectedViewLayout>
  );
}
