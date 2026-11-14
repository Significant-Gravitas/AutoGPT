"use client";

import type { Expert } from "@/app/api/__generated__/models/expert";
import { Text } from "@/components/atoms/Text/Text";
import { ScopeSelect } from "./ScopeSelect";

interface Props {
  scopeExpertID: string | null;
  experts: Expert[];
  onSelect: (expertID: string | null) => void;
}

export function ScopeCard({ scopeExpertID, experts, onSelect }: Props) {
  return (
    <div className="flex flex-col gap-3 rounded-large border border-border-base bg-surface px-4 py-4 shadow-message-table sm:flex-row sm:items-center sm:justify-between">
      <div className="flex min-w-0 flex-col">
        <Text variant="body-medium" as="span">
          Memory scope
        </Text>
        <Text variant="small" as="span" tone="muted">
          Choose whose memory you&apos;re looking at. Experts keep their own.
        </Text>
      </div>
      <div className="w-full sm:w-[280px]">
        <ScopeSelect
          scopeExpertID={scopeExpertID}
          experts={experts}
          onSelect={onSelect}
        />
      </div>
    </div>
  );
}
