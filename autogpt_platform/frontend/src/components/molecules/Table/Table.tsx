import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Input } from "@/components/atoms/Input/Input";
import { Text } from "@/components/atoms/Text/Text";
import {
  Table as TableRoot,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
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
      <TableRoot>
        <TableHeader>
          <TableRow>
            {columns.map((column) => (
              <TableHead key={column} className="text-muted-foreground">
                <Text variant="small-medium" as="span" tone="muted">
                  {formatColumnTitle(column)}
                </Text>
              </TableHead>
            ))}
            {showDeleteColumn && (
              <TableHead className="w-14">
                <span className="sr-only">Actions</span>
              </TableHead>
            )}
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.map((row, rowIndex) => (
            <TableRow key={rowIndex}>
              {columns.map((column) => (
                <TableCell
                  key={`${rowIndex}-${column}`}
                  className={cn("h-auto", readOnly ? "px-4" : "p-2")}
                >
                  {readOnly ? (
                    <Text variant="body" as="span" unmask={false}>
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
                </TableCell>
              ))}
              {showDeleteColumn && (
                <TableCell className="h-auto p-2">
                  <Button
                    variant="icon"
                    size="icon-lg"
                    onClick={() => handleDeleteRow(rowIndex)}
                    aria-label="Delete row"
                    className="text-muted-foreground transition-colors hover:text-destructive"
                  >
                    <Icon icon={Delete02Icon} size={16} aria-hidden />
                  </Button>
                </TableCell>
              )}
            </TableRow>
          ))}
          {showAddButton && (
            <TableRow className="hover:bg-transparent">
              <TableCell
                colSpan={columns.length + (showDeleteColumn ? 1 : 0)}
                className="h-auto p-2"
              >
                <Button
                  variant="outline"
                  size="md"
                  onClick={handleAddRow}
                  leftIcon={<Icon icon={PlusSignIcon} size={16} aria-hidden />}
                  className="w-fit"
                >
                  {addRowLabel}
                </Button>
              </TableCell>
            </TableRow>
          )}
        </TableBody>
      </TableRoot>
    </div>
  );
}

export { type RowData } from "./useTable";
