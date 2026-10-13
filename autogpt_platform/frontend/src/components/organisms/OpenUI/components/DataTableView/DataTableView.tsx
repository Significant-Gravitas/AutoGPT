import type { ComponentRenderProps } from "@openuidev/react-lang";
import type { z } from "zod/v4";
import type { DataTable } from "@/lib/openui/catalog";
import { Input } from "@/components/atoms/Input/Input";
import { useId } from "react";
import { useDataTableView } from "./useDataTableView";

export function DataTableView({
  props,
}: ComponentRenderProps<z.infer<typeof DataTable.props>>) {
  const id = useId();
  const { query, setQuery, sort, toggleSort, visibleRows } = useDataTableView(
    props.rows ?? [],
  );
  return (
    <section className="min-w-0 overflow-hidden rounded-xl border border-zinc-200 bg-white">
      <div className="flex flex-wrap items-center justify-between gap-3 px-5 py-4">
        <h3 className="text-sm font-semibold text-zinc-800">{props.title}</h3>
        <Input
          id={id}
          label={`Filter ${props.title}`}
          hideLabel
          placeholder="Filter results…"
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          size="small"
          className="!h-8 !text-xs"
          wrapperClassName="w-36"
        />
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-left text-xs">
          <thead className="border-y border-zinc-100 bg-zinc-50 text-zinc-500">
            <tr>
              {props.columns?.slice(0, 6).map((column, index) => (
                <th
                  key={`${index}-${column}`}
                  className="px-5 py-3 font-medium"
                  aria-sort={
                    sort.column === index
                      ? sort.ascending
                        ? "ascending"
                        : "descending"
                      : "none"
                  }
                >
                  <button
                    onClick={() => toggleSort(index)}
                    className="whitespace-nowrap rounded-sm text-left focus-visible:outline-purple-500"
                  >
                    {column}
                    <span aria-hidden className="ml-1 text-zinc-400">
                      {sort.column === index
                        ? sort.ascending
                          ? "↑"
                          : "↓"
                        : "↕"}
                    </span>
                  </button>
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-zinc-100">
            {visibleRows.map((row, index) => (
              <tr key={index} className="hover:bg-zinc-50">
                {props.columns?.slice(0, 6).map((column, cellIndex) => (
                  <td
                    key={`${cellIndex}-${column}`}
                    className="px-5 py-3 text-zinc-700 first:font-medium first:text-zinc-900"
                  >
                    {row[cellIndex]}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
        {!visibleRows.length && (
          <p className="p-5 text-sm text-zinc-500">
            No matching results. Try another filter.
          </p>
        )}
      </div>
    </section>
  );
}
