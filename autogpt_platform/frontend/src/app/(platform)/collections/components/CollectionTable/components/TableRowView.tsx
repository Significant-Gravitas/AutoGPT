import type { PageButton } from "@/app/api/__generated__/models/pageButton";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { cn } from "@/lib/utils";
import { Delete02Icon } from "@hugeicons/core-free-icons";
import { RowButtons } from "../../RowButtons/RowButtons";
import type { useRowButtons } from "../../RowButtons/useRowButtons";
import {
  cellText,
  isNumeric,
  type TableColumn,
  type TableRow,
} from "../helpers";
import type { useCollectionTable } from "../useCollectionTable";
import { EditableCell } from "./EditableCell";

export function TableRowView({
  row,
  columns,
  readOnly,
  table,
  buttons,
  runs,
}: {
  row: TableRow;
  columns: TableColumn[];
  readOnly: boolean;
  table: ReturnType<typeof useCollectionTable>;
  buttons: PageButton[];
  runs: ReturnType<typeof useRowButtons>;
}) {
  const rowLabel =
    cellText(columns[0], row.values[columns[0]?.field ?? ""]) || "this row";
  return (
    <tr className="border-t border-separator">
      {columns.map((column) => (
        <td
          key={column.field}
          className={cn(
            "px-4 py-2 align-middle",
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
      {readOnly && !buttons.length ? null : (
        <td className="px-2 py-1.5 align-middle">
          <div className="flex items-center justify-end gap-1.5">
            <RowButtons
              buttons={buttons}
              row={row}
              rowLabel={rowLabel}
              actions={runs}
            />
            {readOnly ? null : (
              <Button
                size="icon-xs"
                variant="ghost"
                aria-label={`Delete ${rowLabel}`}
                onClick={() => table.deleteRow(row)}
              >
                <Icon icon={Delete02Icon} size={16} />
              </Button>
            )}
          </div>
        </td>
      )}
    </tr>
  );
}
