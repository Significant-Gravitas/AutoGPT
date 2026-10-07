import * as React from "react";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Input } from "@/components/atoms/Input/Input";
import { Text } from "@/components/atoms/Text/Text";
import { Delete02Icon, PlusSignIcon } from "@hugeicons/core-free-icons";
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
      <div className="overflow-hidden rounded-xl border border-border bg-card">
        <div className="relative w-full overflow-auto">
          <table className="w-full caption-bottom text-sm">
            <thead>
              <tr className="border-b border-border bg-muted/50">
                {columns.map((column) => (
                  <th
                    key={column}
                    className="h-10 px-3 text-left align-middle text-sm font-medium text-muted-foreground"
                  >
                    {formatColumnTitle(column)}
                  </th>
                ))}
                {showDeleteColumn && (
                  <th className="h-10 w-[50px] px-2 text-left align-middle font-medium text-muted-foreground">
                    <span className="sr-only">Actions</span>
                  </th>
                )}
              </tr>
            </thead>
            <tbody>
              {rows.map((row, rowIndex) => (
                <tr
                  key={rowIndex}
                  className="transition-colors hover:bg-muted/50"
                >
                  {columns.map((column) => (
                    <td
                      key={`${rowIndex}-${column}`}
                      className="p-2 align-middle"
                    >
                      {readOnly ? (
                        <Text
                          variant="body"
                          className="px-3 py-2 text-sm text-foreground"
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
                          size="md"
                          wrapperClassName="mb-0"
                        />
                      )}
                    </td>
                  ))}
                  {showDeleteColumn && (
                    <td className="p-2 align-middle">
                      <Button
                        variant="icon"
                        size="icon-lg"
                        onClick={() => handleDeleteRow(rowIndex)}
                        aria-label="Delete row"
                        className="text-muted-foreground transition-colors hover:text-destructive"
                      >
                        <Icon icon={Delete02Icon} size={16} aria-hidden />
                      </Button>
                    </td>
                  )}
                </tr>
              ))}
              {showAddButton && (
                <tr>
                  <td
                    colSpan={columns.length + (showDeleteColumn ? 1 : 0)}
                    className="p-2 align-middle"
                  >
                    <Button
                      variant="outline"
                      size="md"
                      onClick={handleAddRow}
                      leftIcon={
                        <Icon icon={PlusSignIcon} size={16} aria-hidden />
                      }
                      className="w-fit"
                    >
                      {addRowLabel}
                    </Button>
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

export { type RowData } from "./useTable";
