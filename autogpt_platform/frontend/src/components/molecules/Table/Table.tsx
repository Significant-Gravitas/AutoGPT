import * as React from "react";
import {
  Table as BaseTable,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/__legacy__/ui/table";
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
    <div className={cn("flex flex-col gap-3", className)}>
      <div className="overflow-hidden rounded-xl border border-zinc-200 bg-white">
        <BaseTable>
          <TableHeader>
            <TableRow className="border-b border-zinc-100 bg-zinc-50/50">
              {columns.map((column) => (
                <TableHead
                  key={column}
                  className="h-10 px-3 text-sm font-medium text-zinc-600"
                >
                  {formatColumnTitle(column)}
                </TableHead>
              ))}
              {showDeleteColumn && <TableHead className="w-[50px]" />}
            </TableRow>
          </TableHeader>
          <TableBody>
            {rows.map((row, rowIndex) => (
              <TableRow key={rowIndex} className="border-none">
                {columns.map((column) => (
                  <TableCell key={`${rowIndex}-${column}`} className="p-2">
                    {readOnly ? (
                      <Text
                        variant="body"
                        className="px-3 py-2 text-sm text-zinc-800"
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
                      />
                    )}
                  </TableCell>
                ))}
                {showDeleteColumn && (
                  <TableCell className="p-2">
                    <Button
                      variant="icon"
                      size="icon"
                      onClick={() => handleDeleteRow(rowIndex)}
                      aria-label="Delete row"
                      className="text-zinc-400 transition-colors hover:text-red-500"
                    >
                      <Trash2 className="h-4 w-4" />
                    </Button>
                  </TableCell>
                )}
              </TableRow>
            ))}
            {showAddButton && (
              <TableRow>
                <TableCell
                  colSpan={columns.length + (showDeleteColumn ? 1 : 0)}
                  className="p-2"
                >
                  <Button
                    variant="outline"
                    size="small"
                    onClick={handleAddRow}
                    leftIcon={<Plus className="h-4 w-4" />}
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
