import { useId } from "react";
import { Input } from "@/components/atoms/Input/Input";
import { Button } from "@/components/atoms/Button/Button";
import type { DataColumn } from "@/lib/openui/data-values";
import type { useDashboard } from "./useDashboard";

interface Props {
  columns: DataColumn[];
  dashboard: ReturnType<typeof useDashboard>;
  bound: boolean;
}

export function DashboardFilters({ columns, dashboard, bound }: Props) {
  const id = useId();
  const { filters, rows } = dashboard;
  return (
    <details open className="space-y-3">
      <summary className="min-h-11 cursor-pointer py-3 text-sm">
        Filter this dashboard
      </summary>
      <Input
        id={id}
        label="Search dashboard"
        value={filters.query}
        onChange={(event) =>
          dashboard.setFilters({ ...filters, query: event.target.value })
        }
      />
      <div className="grid gap-3 sm:grid-cols-2">
        {columns.map((column) =>
          column.type === "number" ? (
            <fieldset key={column.key} className="space-y-2">
              <legend className="text-xs">
                {column.label} {column.unit}
              </legend>
              <div className="grid grid-cols-2 gap-2">
                {(["min", "max"] as const).map((bound) => (
                  <Input
                    key={bound}
                    id={`${id}-${column.key}-${bound}`}
                    label={`${column.label} ${bound}`}
                    value={filters.ranges[column.key]?.[bound] ?? ""}
                    inputMode="decimal"
                    onChange={(event) =>
                      dashboard.range(column.key, bound, event.target.value)
                    }
                  />
                ))}
              </div>
            </fieldset>
          ) : (
            <label
              key={column.key}
              className="grid content-start gap-1 text-xs"
            >
              {column.label}
              <select
                className="min-h-11 min-w-0 rounded-md border border-input bg-background px-2 text-sm"
                value={filters.categories[column.key] ?? ""}
                onChange={(event) =>
                  dashboard.category(column.key, event.target.value)
                }
              >
                <option value="">All values</option>
                {[
                  ...new Set(
                    rows.map((row) => String(row.cells[column.key] ?? "")),
                  ),
                ]
                  .filter(Boolean)
                  .sort()
                  .map((value) => (
                    <option key={value}>{value}</option>
                  ))}
              </select>
            </label>
          ),
        )}
      </div>
      {bound && (
        <label className="flex min-h-11 items-center gap-2 text-sm">
          <input
            type="checkbox"
            checked={filters.selectedOnly}
            onChange={(event) =>
              dashboard.setFilters({
                ...filters,
                selectedOnly: event.target.checked,
              })
            }
          />
          Only selected grid rows
        </label>
      )}
      <Button
        type="button"
        size="small"
        variant="ghost"
        onClick={dashboard.reset}
      >
        Reset filters
      </Button>
    </details>
  );
}
