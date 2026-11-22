"use client";

import type { Condition } from "@/app/api/__generated__/models/condition";
import type { Sort } from "@/app/api/__generated__/models/sort";
import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { useState } from "react";
import type { PageButton } from "@/app/api/__generated__/models/pageButton";
import { useRowButtons } from "../RowButtons/useRowButtons";
import { TableRowView } from "./components/TableRowView";
import { NewRowForm } from "./components/NewRowForm";
import { isNumeric, type TableColumn, type TableRow } from "./helpers";
import { useCollectionTable } from "./useCollectionTable";

interface Props {
  collectionID: string;
  columns: TableColumn[];
  rows: TableRow[];
  total: number;
  filter?: Condition[];
  sort?: Sort | null;
  readOnly: boolean;
  buttons?: PageButton[];
  /** Absent where nothing runs, as in a chat preview. */
  onRunStarted?(libraryAgentID: string, runID: string): void;
}

export function CollectionTable({
  collectionID,
  columns,
  rows,
  total,
  filter = [],
  sort = null,
  readOnly,
  buttons = [],
  onRunStarted,
}: Props) {
  const table = useCollectionTable({
    collectionID,
    columns,
    rows,
    total,
    filter,
    sort,
  });
  const [adding, setAdding] = useState(false);
  const runs = useRowButtons({
    onRunStarted,
    onRunFinished: () => void table.refresh(),
  });
  const runnable = onRunStarted ? buttons : [];
  const hasActions = !readOnly || runnable.length > 0;

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-center gap-2">
        <Input
          id={`search-${collectionID}`}
          label="Search rows"
          hideLabel
          size="small"
          placeholder="Search"
          wrapperClassName="!mb-0 w-48"
          value={table.search}
          onChange={(e) => table.setSearch(e.target.value)}
        />
        {table.chipField?.options?.map((option) => {
          const on = table.chips.includes(option);
          return (
            <Button
              key={option}
              size="xs"
              variant={on ? "secondary" : "ghost"}
              aria-pressed={on}
              onClick={() => table.toggleChip(option)}
            >
              {option}
            </Button>
          );
        })}
        {readOnly || adding ? null : (
          <Button
            size="small"
            variant="outline"
            className="ml-auto"
            onClick={() => setAdding(true)}
          >
            Add row
          </Button>
        )}
      </div>
      {adding ? (
        <NewRowForm
          columns={columns}
          onAdd={table.addRow}
          onCancel={() => setAdding(false)}
        />
      ) : null}
      <div className="overflow-x-auto rounded-xl bg-surface smooth-shadow-ring-sm">
        <table className="w-full border-collapse text-left">
          <thead className="bg-gradient-to-b from-surface-muted to-surface-hover/50">
            <tr>
              {columns.map((column) => (
                <th
                  key={column.field}
                  scope="col"
                  className={cn(
                    "px-4 py-2.5 font-normal whitespace-nowrap",
                    isNumeric(column) && "text-right",
                  )}
                >
                  <Text as="span" variant="small-medium" tone="secondary">
                    {column.field}
                  </Text>
                </th>
              ))}
              {hasActions ? (
                <th scope="col" className="w-10 px-2">
                  <span className="sr-only">Actions</span>
                </th>
              ) : null}
            </tr>
          </thead>
          <tbody>
            {table.rows.map((row) => (
              <TableRowView
                key={row.id}
                row={row}
                columns={columns}
                readOnly={readOnly}
                table={table}
                buttons={runnable}
                runs={runs}
              />
            ))}
          </tbody>
        </table>
        {table.rows.length ? null : (
          <div className="px-4 py-6">
            <Text variant="body" tone="secondary">
              {table.isNarrowing ? "Looking…" : "No rows."}
            </Text>
          </div>
        )}
      </div>
      {table.total > table.rows.length ? (
        <Text variant="small" tone="secondary">
          {table.rows.length} of {table.total.toLocaleString("en-US")} rows
        </Text>
      ) : null}
    </div>
  );
}
