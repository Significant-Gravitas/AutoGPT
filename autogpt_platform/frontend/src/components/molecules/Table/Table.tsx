import * as React from "react";
import {
  Table as BaseTable,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "./components/TablePrimitives";
import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Text } from "@/components/atoms/Text/Text";
import { Plus, Trash2 } from "lucide-react";
import { cn } from "@/lib/utils";
import { useTable, RowData } from "./useTable";
import { formatColumnTitle, formatPlaceholder } from "./helpers";

export interface TableProps {
  columns: string[];
  defaultValues?: RowData[];
  onChange?: (rows: RowData[]) => void;
  allowAddRow?: boolean;
  allowDeleteRow?: boolean;
  addRowLabel?: string;
  className?: string;
  readOnly?: boolean;
}

export function Table({
  columns,
  defaultValues,
  onChange,
  allowAddRow = true,
  allowDeleteRow = true,
  addRowLabel = "Add row",
  className,
  readOnly = false,
}: TableProps) {
  const { rows, handleAddRow, handleDeleteRow, handleCellChange } = useTable({
    columns,
    defaultValues,
    onChange,
  });

  const showDeleteColumn = allowDeleteRow && !readOnly;
  const showAddButton = allowAddRow && !readOnly;

  return (
    <div data-slot="table" className={cn("flex flex-col gap-3", className)}>
      <div
        data-slot="table-frame"
        className="overflow-hidden rounded-xl border border-border-base bg-surface"
      >
        <BaseTable data-slot="table-element">
          <TableHeader data-slot="table-header">
            <TableRow
              data-slot="table-header-row"
              className="border-b border-separator bg-surface-muted/50"
            >
              {columns.map((column) => (
                <TableHead
                  key={column}
                  data-slot="table-head"
                  className="h-10 px-3 text-body font-medium leading-5 text-foreground-secondary"
                >
                  {formatColumnTitle(column)}
                </TableHead>
              ))}
              {showDeleteColumn && (
                <TableHead data-slot="table-head" className="w-[50px]" />
              )}
            </TableRow>
          </TableHeader>
          <TableBody data-slot="table-body">
            {rows.map((row, rowIndex) => (
              <TableRow
                key={rowIndex}
                data-slot="table-row"
                className="border-none"
              >
                {columns.map((column) => (
                  <TableCell
                    key={`${rowIndex}-${column}`}
                    data-slot="table-cell"
                    className="p-2"
                  >
                    {readOnly ? (
                      <Text
                        variant="body"
                        data-slot="table-cell-text"
                        className="px-3 py-2 text-body leading-5 text-foreground-emphasis"
                      >
                        {row[column] || "-"}
                      </Text>
                    ) : (
                      <Input
                        id={`table-${rowIndex}-${column}`}
                        label={formatColumnTitle(column)}
                        hideLabel
                        value={row[column] ?? ""}
                        onChange={(e) =>
                          handleCellChange(rowIndex, column, e.target.value)
                        }
                        placeholder={formatPlaceholder(column)}
                        size="small"
                        wrapperClassName="mb-0"
                        data-slot="table-cell-input"
                      />
                    )}
                  </TableCell>
                ))}
                {showDeleteColumn && (
                  <TableCell data-slot="table-cell" className="p-2">
                    <Button
                      variant="icon"
                      size="icon"
                      onClick={() => handleDeleteRow(rowIndex)}
                      aria-label="Delete row"
                      data-slot="table-delete-row"
                      className="text-foreground-subtle transition-colors hover:text-error"
                    >
                      <Trash2
                        data-slot="table-delete-row-icon"
                        className="h-4 w-4"
                      />
                    </Button>
                  </TableCell>
                )}
              </TableRow>
            ))}
            {showAddButton && (
              <TableRow data-slot="table-row">
                <TableCell
                  data-slot="table-cell"
                  colSpan={columns.length + (showDeleteColumn ? 1 : 0)}
                  className="p-2"
                >
                  <Button
                    variant="outline"
                    size="small"
                    onClick={handleAddRow}
                    data-slot="table-add-row"
                    leftIcon={
                      <Plus
                        data-slot="table-add-row-icon"
                        className="h-4 w-4"
                      />
                    }
                    className="w-fit"
                  >
                    {addRowLabel}
                  </Button>
                </TableCell>
              </TableRow>
            )}
          </TableBody>
        </BaseTable>
      </div>
    </div>
  );
}

export { type RowData } from "./useTable";
