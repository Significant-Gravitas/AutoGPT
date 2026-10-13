import type { DataColumn, DataRow, CellValue } from "@/lib/openui/data-values";
import { Button } from "@/components/atoms/Button/Button";
import { CellEditor } from "../CellEditor";

interface Props {
  row: DataRow;
  columns: DataColumn[];
  editable: boolean;
  selectable: boolean;
  selected: boolean;
  removable: boolean;
  disabled: boolean;
  onEdit: (key: string, value: CellValue) => void;
  onSelect: () => void;
  onRemove: () => void;
}

export function DataGridRow({
  row,
  columns,
  editable,
  selectable,
  selected,
  removable,
  disabled,
  onEdit,
  onSelect,
  onRemove,
}: Props) {
  return (
    <tr className="border-b border-border align-top">
      {selectable && (
        <td className="p-2">
          <input
            type="checkbox"
            aria-label={`Select ${row.id}`}
            checked={selected}
            disabled={disabled}
            onChange={onSelect}
            className="m-3 h-4 w-4"
          />
        </td>
      )}
      {columns.map((column) => (
        <td key={column.key} className="min-w-36 p-2">
          {editable && !column.calculation ? (
            <CellEditor
              column={column}
              value={row.cells[column.key] ?? null}
              label={`${column.label} · ${row.id}`}
              disabled={disabled}
              onChange={(value) => onEdit(column.key, value)}
            />
          ) : (
            <span className="break-words text-sm">
              {row.cells[column.key] == null
                ? "Unknown"
                : String(row.cells[column.key])}
            </span>
          )}
        </td>
      ))}
      <td className="p-2">
        {row.source && (
          <a
            href={row.source.url}
            target="_blank"
            rel="noopener noreferrer"
            className="text-xs text-primary underline"
          >
            {row.source.label}
          </a>
        )}
        {removable && (
          <Button
            type="button"
            variant="ghost"
            size="small"
            disabled={disabled}
            onClick={onRemove}
          >
            Remove {row.id}
          </Button>
        )}
      </td>
    </tr>
  );
}
