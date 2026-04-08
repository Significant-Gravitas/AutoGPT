import type { CostLogRow } from "@/app/api/__generated__/models/costLogRow";
import type { Pagination } from "@/app/api/__generated__/models/pagination";
import { formatDuration, formatMicrodollars, formatTokens } from "../helpers";
import { TrackingBadge } from "./TrackingBadge";

function formatLogDate(value: unknown): string {
  if (value instanceof Date) return value.toLocaleString();
  if (typeof value === "string" || typeof value === "number")
    return new Date(value).toLocaleString();
  return "-";
}

interface Props {
  logs: CostLogRow[];
  pagination: Pagination | null;
  onPageChange: (page: number) => void;
}

function LogsTable({ logs, pagination, onPageChange }: Props) {
  return (
    <div className="flex flex-col gap-4">
      <div className="overflow-x-auto">
        <table className="w-full text-left text-sm">
          <thead className="border-b text-xs uppercase text-muted-foreground">
            <tr>
              <th scope="col" className="px-3 py-3">
                Time
              </th>
              <th scope="col" className="px-3 py-3">
                User
              </th>
              <th scope="col" className="px-3 py-3">
                Block
              </th>
              <th scope="col" className="px-3 py-3">
                Provider
              </th>
              <th scope="col" className="px-3 py-3">
                Type
              </th>
              <th scope="col" className="px-3 py-3">
                Model
              </th>
              <th scope="col" className="px-3 py-3 text-right">
                Cost
              </th>
              <th scope="col" className="px-3 py-3 text-right">
                Tokens
              </th>
              <th scope="col" className="px-3 py-3 text-right">
                Duration
              </th>
              <th scope="col" className="px-3 py-3">
                Execution
              </th>
            </tr>
          </thead>
          <tbody>
            {logs.map((log) => (
              <tr key={log.id} className="border-b hover:bg-muted">
                <td className="whitespace-nowrap px-3 py-2 text-xs">
                  {formatLogDate(log.created_at)}
                </td>
                <td className="px-3 py-2 text-xs">
                  {log.email ||
                    (log.user_id ? String(log.user_id).slice(0, 8) : "-")}
                </td>
                <td className="px-3 py-2 text-xs font-medium">
                  {log.block_name}
                </td>
                <td className="px-3 py-2 text-xs">{log.provider}</td>
                <td className="px-3 py-2 text-xs">
                  <TrackingBadge trackingType={log.tracking_type} />
                </td>
                <td className="px-3 py-2 text-xs">{log.model || "-"}</td>
                <td className="px-3 py-2 text-right text-xs">
                  {log.cost_microdollars != null
                    ? formatMicrodollars(Number(log.cost_microdollars))
                    : "-"}
                </td>
                <td className="px-3 py-2 text-right text-xs">
                  {log.input_tokens != null || log.output_tokens != null
                    ? `${formatTokens(Number(log.input_tokens ?? 0))} / ${formatTokens(Number(log.output_tokens ?? 0))}`
                    : "-"}
                </td>
                <td className="px-3 py-2 text-right text-xs">
                  {log.duration != null
                    ? formatDuration(Number(log.duration))
                    : "-"}
                </td>
                <td className="px-3 py-2 text-xs text-muted-foreground">
                  {log.graph_exec_id
                    ? String(log.graph_exec_id).slice(0, 8)
                    : "-"}
                </td>
              </tr>
            ))}
            {logs.length === 0 && (
              <tr>
                <td
                  colSpan={10}
                  className="px-4 py-8 text-center text-muted-foreground"
                >
                  No logs found
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {pagination && pagination.total_pages > 1 && (
        <div className="flex items-center justify-between px-4">
          <span className="text-sm text-muted-foreground">
            Page {pagination.current_page} of {pagination.total_pages} (
            {pagination.total_items} total)
          </span>
          <div className="flex gap-2">
            <button
              disabled={pagination.current_page <= 1}
              onClick={() => onPageChange(pagination.current_page - 1)}
              className="rounded border px-3 py-1 text-sm disabled:opacity-50"
            >
              Previous
            </button>
            <button
              disabled={pagination.current_page >= pagination.total_pages}
              onClick={() => onPageChange(pagination.current_page + 1)}
              className="rounded border px-3 py-1 text-sm disabled:opacity-50"
            >
              Next
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export { LogsTable };
