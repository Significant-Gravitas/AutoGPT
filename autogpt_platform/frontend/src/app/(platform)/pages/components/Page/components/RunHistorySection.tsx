import type { HistoryRow } from "@/app/api/__generated__/models/historyRow";
import type { PageRunHistory } from "@/app/api/__generated__/models/pageRunHistory";
import { Text } from "@/components/atoms/Text/Text";
import { RunStatusBadge } from "@/components/molecules/RunStatusBadge/RunStatusBadge";
import { formatDate } from "@/lib/utils/time";
import NextLink from "next/link";
import { useState } from "react";
import { libraryRunHref } from "../helpers";

interface Props {
  section: PageRunHistory;
}

// Column -1 is the run's start time.
interface Sort {
  column: number;
  descending: boolean;
}

export function RunHistorySection({ section }: Props) {
  const [sort, setSort] = useState<Sort>({ column: -1, descending: true });
  if (!section.rows.length) {
    return (
      <Text variant="body" tone="secondary">
        No runs yet.
      </Text>
    );
  }
  const rows = sortedRows(section.rows, sort);
  const headers = ["When", ...section.columns];

  function toggle(column: number) {
    setSort((s) => ({
      column,
      descending: s.column === column ? !s.descending : column === -1,
    }));
  }

  return (
    <div className="overflow-x-auto rounded-xl bg-surface smooth-shadow-ring-sm">
      <table className="w-full border-collapse text-left">
        <thead className="bg-surface-muted">
          <tr>
            {headers.map((header, position) => {
              const column = position - 1;
              const active = sort.column === column;
              return (
                <th
                  key={position}
                  scope="col"
                  aria-sort={
                    active
                      ? sort.descending
                        ? "descending"
                        : "ascending"
                      : "none"
                  }
                  className="px-4 py-2.5 font-normal whitespace-nowrap"
                >
                  <button type="button" onClick={() => toggle(column)}>
                    <Text as="span" variant="small-medium" tone="secondary">
                      {header}
                      {active ? (sort.descending ? " ↓" : " ↑") : ""}
                    </Text>
                  </button>
                </th>
              );
            })}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.run_id} className="border-t border-separator">
              <td className="px-4 py-2 whitespace-nowrap">
                <NextLink
                  href={libraryRunHref(section.library_agent_id, row.run_id)}
                  className="flex items-center gap-2"
                >
                  <RunStatusBadge status={row.status} />
                  <Text as="span" variant="body">
                    {row.started_at
                      ? formatDate(new Date(row.started_at))
                      : "Queued"}
                  </Text>
                </NextLink>
              </td>
              {row.cells.map((cell, index) => (
                <td key={index} className="px-4 py-2 align-top">
                  <Text as="span" variant="body" className="break-words">
                    {cell ?? "–"}
                  </Text>
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function sortedRows(rows: HistoryRow[], sort: Sort) {
  const value = (row: HistoryRow) =>
    sort.column === -1
      ? String(row.started_at ?? "")
      : (row.cells[sort.column] ?? "");
  const sign = sort.descending ? -1 : 1;
  return [...rows].sort((a, b) => {
    const [x, y] = [value(a), value(b)];
    const [nx, ny] = [
      Number(x.replace(/[$,%]/g, "")),
      Number(y.replace(/[$,%]/g, "")),
    ];
    if (sort.column !== -1 && x && y && !isNaN(nx) && !isNaN(ny)) {
      return sign * (nx - ny);
    }
    return sign * x.localeCompare(y);
  });
}
