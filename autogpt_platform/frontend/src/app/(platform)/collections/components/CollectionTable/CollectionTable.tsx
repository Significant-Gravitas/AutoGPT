"use client";

import type { Condition } from "@/app/api/__generated__/models/condition";
import type { Sort } from "@/app/api/__generated__/models/sort";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Input } from "@/components/atoms/Input/Input";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { Delete02Icon } from "@hugeicons/core-free-icons";
import { useState } from "react";
import { EditableCell } from "./components/EditableCell";
import { NewRowForm } from "./components/NewRowForm";
import {
  cellText,
  isNumeric,
  type TableColumn,
  type TableRow,
} from "./helpers";
import { useCollectionTable } from "./useCollectionTable";

interface Props {
  collectionID: string;
  columns: TableColumn[];
  rows: TableRow[];
  total: number;
  filter?: Condition[];
  sort?: Sort | null;
  readOnly: boolean;
}

export function CollectionTable({
  collectionID,
  columns,
  rows,
  total,
  filter = [],
  sort = null,
  readOnly,
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

  return (
    <div className="flex flex-col gap-3">
      <div className="flex flex-wrap items-end gap-2">
        <Input
          id={`search-${collectionID}`}
          label="Search rows"
          hideLabel
          size="small"
          placeholder="Search"
          wrapperClassName="w-48"
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
              {readOnly ? null : (
                <th scope="col" className="w-10 px-2">
                  <span className="sr-only">Actions</span>
                </th>
              )}
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

function TableRowView({
  row,
  columns,
  readOnly,
  table,
}: {
  row: TableRow;
  columns: TableColumn[];
  readOnly: boolean;
  table: ReturnType<typeof useCollectionTable>;
}) {
  const rowLabel =
    cellText(columns[0], row.values[columns[0]?.field ?? ""]) || "this row";
  return (
    <tr className="border-t border-separator">
      {columns.map((column) => (
        <td
          key={column.field}
          className={cn(
            "px-4 py-2 align-top",
            isNumeric(column) ? "text-right" : "min-w-32",
          )}
        >
          <EditableCell
            column={column}
            row={row}
            rowLabel={rowLabel}
            readOnly={readOnly}
            onSave={(value) => table.saveCell(row, column.field, value)}
            onAccept={() => table.acceptPending(row, column.field)}
          />
        </td>
      ))}
      {readOnly ? null : (
        <td className="px-2 py-1.5 text-right align-top">
          <Button
            size="icon-xs"
            variant="ghost"
            aria-label={`Delete ${rowLabel}`}
            onClick={() => table.deleteRow(row)}
          >
            <Icon icon={Delete02Icon} size={16} />
          </Button>
        </td>
      )}
    </tr>
  );
}
