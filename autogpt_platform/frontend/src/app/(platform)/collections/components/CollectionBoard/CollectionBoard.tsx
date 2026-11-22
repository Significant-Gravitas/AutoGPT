"use client";

import type { PageCollectionBoard } from "@/app/api/__generated__/models/pageCollectionBoard";
import { Select } from "@/components/atoms/Select/Select";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { cellText } from "../CollectionTable/helpers";
import { RowCard } from "../RowCard/RowCard";
import { NO_VALUE, useCollectionBoard } from "./useCollectionBoard";

interface Props {
  section: PageCollectionBoard;
  canEdit: boolean;
  onRunStarted?(libraryAgentID: string, runID: string): void;
}

export function CollectionBoard({ section, canEdit, onRunStarted }: Props) {
  const board = useCollectionBoard({ section, canEdit, onRunStarted });
  const options = section.columns
    .map((column) => column.option)
    .filter((option): option is string => option !== null);

  return (
    <div className="flex gap-3 overflow-x-auto pb-1">
      {section.columns.map((column) => {
        const name = column.option ?? NO_VALUE;
        return (
          <section
            key={name}
            aria-label={name}
            className={cn(
              "flex min-w-48 flex-1 flex-col gap-2 rounded-2xl bg-surface-muted p-2",
              board.dropTarget === name && "ring-2 ring-focus",
            )}
            onDragOver={(e) => board.onDragOver(e, name)}
            onDragLeave={() => board.onDragLeave(name)}
            onDrop={() => board.onDrop(column.option)}
          >
            <div className="flex items-center justify-between px-1">
              <Text variant="small-medium">{name}</Text>
              <Text variant="small" tone="secondary" tabular>
                {column.total}
              </Text>
            </div>
            {column.rows.map((row) => (
              <div
                key={row.id}
                draggable={!board.readOnly}
                onDragStart={() => board.onDragStart(row.id)}
                onDragEnd={board.onDragEnd}
                className={cn(!board.readOnly && "cursor-grab")}
              >
                <RowCard
                  row={row}
                  title={section.card_title}
                  fields={section.card_fields}
                  buttons={section.buttons ?? []}
                  actions={board.buttons}
                  canRun={canEdit}
                  footer={
                    board.readOnly ? null : (
                      <Select
                        id={`move-${row.id}`}
                        label={`Move ${cellText(section.card_title, row.values[section.card_title.field]) || "card"}`}
                        hideLabel
                        size="small"
                        wrapperClassName="!mb-0"
                        value={column.option ?? ""}
                        placeholder={NO_VALUE}
                        options={options.map((o) => ({ value: o, label: o }))}
                        onValueChange={(next) => board.move(row, next)}
                      />
                    )
                  }
                />
              </div>
            ))}
            {column.total > column.rows.length ? (
              <div className="px-1">
                <Text variant="small" tone="secondary">
                  {column.total - column.rows.length} more
                </Text>
              </div>
            ) : null}
          </section>
        );
      })}
    </div>
  );
}
