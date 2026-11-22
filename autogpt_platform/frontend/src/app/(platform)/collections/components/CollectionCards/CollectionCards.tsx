"use client";

import type { PageCollectionCards } from "@/app/api/__generated__/models/pageCollectionCards";
import { Text } from "@/components/atoms/Text/Text";
import { useRowWrites } from "../CollectionTable/useRowWrites";
import { RowCard } from "../RowCard/RowCard";
import { useRowButtons } from "../RowButtons/useRowButtons";

interface Props {
  section: PageCollectionCards;
  canRun: boolean;
  onRunStarted?(libraryAgentID: string, runID: string): void;
}

export function CollectionCards({ section, canRun, onRunStarted }: Props) {
  const writes = useRowWrites(section.collection_id);
  const buttons = useRowButtons({
    onRunStarted,
    onRunFinished: () => void writes.refresh(),
  });

  if (!section.rows.length) {
    return (
      <Text variant="body" tone="secondary">
        No rows.
      </Text>
    );
  }
  return (
    <div className="flex flex-col gap-2">
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {section.rows.map((row) => (
          <RowCard
            key={row.id}
            row={row}
            title={section.card_title}
            fields={section.card_fields}
            buttons={section.buttons ?? []}
            actions={buttons}
            canRun={canRun}
          />
        ))}
      </div>
      {section.total > section.rows.length ? (
        <Text variant="small" tone="secondary">
          {section.rows.length} of {section.total.toLocaleString("en-US")} rows
        </Text>
      ) : null}
    </div>
  );
}
