import { useId } from "react";
import type { ComponentRenderProps } from "@openuidev/react-lang";
import type { z } from "zod/v4";
import type { DataGrid } from "@/lib/openui/catalog-data";
import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { csvText, downloadText } from "@/lib/openui/download";
import { useOpenUIDisabled } from "../../interactionContext";
import { ControlValidity } from "../ControlValidity";
import { useDataGrid } from "./useDataGrid";
import { DataGridRow } from "./DataGridRow";

export function DataGridView({
  props,
}: ComponentRenderProps<z.infer<typeof DataGrid.props>>) {
  const id = useId();
  const grid = useDataGrid(props);
  const disabled = useOpenUIDisabled();
  return (
    <fieldset disabled={disabled} className="min-w-0 space-y-3">
      <legend className="text-sm font-semibold">{props.title}</legend>
      <div className="flex flex-wrap items-end gap-2">
        <Input
          id={id}
          label={`Search ${props.title}`}
          value={grid.query}
          onChange={(event) => grid.setQuery(event.target.value)}
        />
        <Button
          type="button"
          variant="secondary"
          size="small"
          disabled={!!grid.errors.length}
          onClick={() =>
            downloadText(
              `${props.name}.csv`,
              csvText([
                props.columns.map((column) => column.label),
                ...grid.rows.map((row) =>
                  props.columns.map((column) => row.cells[column.key]),
                ),
              ]),
              "text/csv;charset=utf-8",
            )
          }
        >
          Export CSV
        </Button>
        {props.editable && props.allowRows && (
          <Button
            type="button"
            variant="secondary"
            size="small"
            onClick={grid.add}
            disabled={grid.rows.length >= 100}
          >
            Add row
          </Button>
        )}
      </div>
      <ControlValidity label={props.title} error={grid.errors[0] ?? ""} />
      <div className="overflow-x-auto">
        <table className="w-full text-left" aria-label={props.title}>
          <thead>
            <tr>
              {props.selection !== "none" && (
                <th className="p-2 text-xs">Select</th>
              )}
              {props.columns.map((column) => (
                <th
                  key={column.key}
                  className="p-2 text-xs"
                  aria-sort={
                    grid.sort.column === column.key
                      ? grid.sort.ascending
                        ? "ascending"
                        : "descending"
                      : "none"
                  }
                >
                  <button
                    type="button"
                    className="min-h-11 text-left"
                    onClick={() => grid.toggleSort(column.key)}
                  >
                    {column.label}{" "}
                    {grid.sort.column === column.key
                      ? grid.sort.ascending
                        ? "↑"
                        : "↓"
                      : "↕"}
                  </button>
                </th>
              ))}
              <th className="p-2 text-xs">Details</th>
            </tr>
          </thead>
          <tbody>
            {grid.visible.map((row) => (
              <DataGridRow
                key={row.id}
                row={row}
                columns={props.columns}
                editable={props.editable}
                selectable={props.selection !== "none"}
                selected={grid.selected.includes(row.id)}
                removable={props.editable && !!props.allowRows}
                disabled={disabled}
                onEdit={(key, value) => grid.edit(row.id, key, value)}
                onSelect={() => grid.select(row.id)}
                onRemove={() => grid.remove(row.id)}
              />
            ))}
          </tbody>
        </table>
      </div>
      {!grid.visible.length && (
        <p className="text-sm text-muted-foreground">No matching rows.</p>
      )}
      <div className="flex flex-wrap items-center gap-3 text-xs">
        <span aria-live="polite">
          {grid.total} matching rows · {grid.selected.length} selected
        </span>
        <Button
          type="button"
          size="small"
          variant="ghost"
          disabled={!grid.page}
          onClick={() => grid.setPage(grid.page - 1)}
        >
          Previous
        </Button>
        <span>
          Page {grid.page + 1} of {Math.max(1, Math.ceil(grid.total / 10))}
        </span>
        <Button
          type="button"
          size="small"
          variant="ghost"
          disabled={(grid.page + 1) * 10 >= grid.total}
          onClick={() => grid.setPage(grid.page + 1)}
        >
          Next
        </Button>
      </div>
    </fieldset>
  );
}
