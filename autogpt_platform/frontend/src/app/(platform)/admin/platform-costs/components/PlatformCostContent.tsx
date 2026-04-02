"use client";

import { useEffect, useState } from "react";
import type {
  PlatformCostDashboard,
  CostLogRow,
  Pagination,
} from "@/lib/autogpt-server-api";
import { getPlatformCostDashboard, getPlatformCostLogs } from "../actions";
import { useRouter, useSearchParams } from "next/navigation";

interface Props {
  searchParams: {
    start?: string;
    end?: string;
    provider?: string;
    user_id?: string;
    page?: string;
    tab?: string;
  };
}

function formatMicrodollars(microdollars: number) {
  return `$${(microdollars / 1_000_000).toFixed(4)}`;
}

function formatTokens(tokens: number) {
  if (tokens >= 1_000_000) return `${(tokens / 1_000_000).toFixed(1)}M`;
  if (tokens >= 1_000) return `${(tokens / 1_000).toFixed(1)}K`;
  return tokens.toString();
}

function PlatformCostContent({ searchParams }: Props) {
  const router = useRouter();
  const urlParams = useSearchParams();
  const [dashboard, setDashboard] = useState<PlatformCostDashboard | null>(
    null,
  );
  const [logs, setLogs] = useState<CostLogRow[]>([]);
  const [pagination, setPagination] = useState<Pagination | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const tab = urlParams.get("tab") || searchParams.tab || "overview";
  const page = parseInt(urlParams.get("page") || searchParams.page || "1", 10);
  const startDate = urlParams.get("start") || searchParams.start || "";
  const endDate = urlParams.get("end") || searchParams.end || "";
  const providerFilter =
    urlParams.get("provider") || searchParams.provider || "";
  const userFilter = urlParams.get("user_id") || searchParams.user_id || "";

  const [startInput, setStartInput] = useState(startDate);
  const [endInput, setEndInput] = useState(endDate);
  const [providerInput, setProviderInput] = useState(providerFilter);
  const [userInput, setUserInput] = useState(userFilter);

  useEffect(() => {
    async function load() {
      setLoading(true);
      setError(null);
      const filters: Record<string, string> = {};
      if (startDate) filters.start = startDate;
      if (endDate) filters.end = endDate;
      if (providerFilter) filters.provider = providerFilter;
      if (userFilter) filters.user_id = userFilter;

      try {
        const [dashData, logsData] = await Promise.all([
          getPlatformCostDashboard(filters),
          getPlatformCostLogs({ ...filters, page, page_size: 50 }),
        ]);
        setDashboard(dashData);
        setLogs(logsData.logs);
        setPagination(logsData.pagination);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Failed to load cost data");
      } finally {
        setLoading(false);
      }
    }
    load();
  }, [startDate, endDate, providerFilter, userFilter, page]);

  function updateUrl(overrides: Record<string, string>) {
    const params = new URLSearchParams(urlParams.toString());
    for (const [k, v] of Object.entries(overrides)) {
      if (v) params.set(k, v);
      else params.delete(k);
    }
    router.push(`/admin/platform-costs?${params.toString()}`);
  }

  function handleFilter() {
    updateUrl({
      start: startInput,
      end: endInput,
      provider: providerInput,
      user_id: userInput,
      page: "1",
    });
  }

  return (
    <div className="flex flex-col gap-6">
      {/* Filters */}
      <div className="flex flex-wrap items-end gap-3 rounded-lg border p-4">
        <div className="flex flex-col gap-1">
          <label className="text-sm text-muted-foreground">Start Date</label>
          <input
            type="datetime-local"
            className="rounded border px-3 py-1.5 text-sm"
            value={startInput}
            onChange={(e) => setStartInput(e.target.value)}
          />
        </div>
        <div className="flex flex-col gap-1">
          <label className="text-sm text-muted-foreground">End Date</label>
          <input
            type="datetime-local"
            className="rounded border px-3 py-1.5 text-sm"
            value={endInput}
            onChange={(e) => setEndInput(e.target.value)}
          />
        </div>
        <div className="flex flex-col gap-1">
          <label className="text-sm text-muted-foreground">Provider</label>
          <input
            type="text"
            placeholder="e.g. openai"
            className="rounded border px-3 py-1.5 text-sm"
            value={providerInput}
            onChange={(e) => setProviderInput(e.target.value)}
          />
        </div>
        <div className="flex flex-col gap-1">
          <label className="text-sm text-muted-foreground">User ID</label>
          <input
            type="text"
            placeholder="Filter by user"
            className="rounded border px-3 py-1.5 text-sm"
            value={userInput}
            onChange={(e) => setUserInput(e.target.value)}
          />
        </div>
        <button
          onClick={handleFilter}
          className="rounded bg-primary px-4 py-1.5 text-sm text-primary-foreground hover:bg-primary/90"
        >
          Apply
        </button>
      </div>

      {error && (
        <div className="rounded-lg border border-red-300 bg-red-50 p-4 text-sm text-red-700 dark:border-red-700 dark:bg-red-900/20 dark:text-red-400">
          {error}
        </div>
      )}

      {loading ? (
        <div className="py-10 text-center text-muted-foreground">
          Loading...
        </div>
      ) : (
        <>
          {/* Summary cards */}
          {dashboard && (
            <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
              <SummaryCard
                label="Total Cost"
                value={formatMicrodollars(dashboard.total_cost_microdollars)}
              />
              <SummaryCard
                label="Total Requests"
                value={dashboard.total_requests.toLocaleString()}
              />
              <SummaryCard
                label="Providers"
                value={dashboard.by_provider.length.toString()}
              />
              <SummaryCard
                label="Active Users"
                value={dashboard.total_users.toLocaleString()}
              />
            </div>
          )}

          {/* Tabs */}
          <div className="flex gap-2 border-b">
            {["overview", "by-user", "logs"].map((t) => (
              <button
                key={t}
                onClick={() => updateUrl({ tab: t, page: "1" })}
                className={`px-4 py-2 text-sm font-medium ${
                  tab === t
                    ? "border-b-2 border-blue-600 text-blue-600"
                    : "text-muted-foreground hover:text-foreground"
                }`}
              >
                {t === "overview"
                  ? "By Provider"
                  : t === "by-user"
                    ? "By User"
                    : "Raw Logs"}
              </button>
            ))}
          </div>

          {/* Tab content */}
          {tab === "overview" && dashboard && (
            <ProviderTable data={dashboard.by_provider} />
          )}
          {tab === "by-user" && dashboard && (
            <UserTable data={dashboard.by_user} />
          )}
          {tab === "logs" && (
            <LogsTable
              logs={logs}
              pagination={pagination}
              onPageChange={(p) => updateUrl({ page: p.toString() })}
            />
          )}
        </>
      )}
    </div>
  );
}

function SummaryCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border p-4">
      <div className="text-sm text-muted-foreground">{label}</div>
      <div className="text-2xl font-bold">{value}</div>
    </div>
  );
}

function ProviderTable({
  data,
}: {
  data: PlatformCostDashboard["by_provider"];
}) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left text-sm">
        <thead className="border-b text-xs uppercase text-muted-foreground">
          <tr>
            <th className="px-4 py-3">Provider</th>
            <th className="px-4 py-3 text-right">Total Cost</th>
            <th className="px-4 py-3 text-right">Requests</th>
            <th className="px-4 py-3 text-right">Input Tokens</th>
            <th className="px-4 py-3 text-right">Output Tokens</th>
          </tr>
        </thead>
        <tbody>
          {data.map((row) => (
            <tr key={row.provider} className="border-b hover:bg-muted">
              <td className="px-4 py-3 font-medium">{row.provider}</td>
              <td className="px-4 py-3 text-right">
                {formatMicrodollars(row.total_cost_microdollars)}
              </td>
              <td className="px-4 py-3 text-right">
                {row.request_count.toLocaleString()}
              </td>
              <td className="px-4 py-3 text-right">
                {formatTokens(row.total_input_tokens)}
              </td>
              <td className="px-4 py-3 text-right">
                {formatTokens(row.total_output_tokens)}
              </td>
            </tr>
          ))}
          {data.length === 0 && (
            <tr>
              <td
                colSpan={5}
                className="px-4 py-8 text-center text-muted-foreground"
              >
                No cost data yet
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}

function UserTable({ data }: { data: PlatformCostDashboard["by_user"] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left text-sm">
        <thead className="border-b text-xs uppercase text-muted-foreground">
          <tr>
            <th className="px-4 py-3">User</th>
            <th className="px-4 py-3 text-right">Total Cost</th>
            <th className="px-4 py-3 text-right">Requests</th>
            <th className="px-4 py-3 text-right">Input Tokens</th>
            <th className="px-4 py-3 text-right">Output Tokens</th>
          </tr>
        </thead>
        <tbody>
          {data.map((row) => (
            <tr key={row.user_id} className="border-b hover:bg-muted">
              <td className="px-4 py-3">
                <div className="font-medium">{row.email || "Unknown"}</div>
                <div className="text-xs text-muted-foreground">
                  {row.user_id}
                </div>
              </td>
              <td className="px-4 py-3 text-right">
                {formatMicrodollars(row.total_cost_microdollars)}
              </td>
              <td className="px-4 py-3 text-right">
                {row.request_count.toLocaleString()}
              </td>
              <td className="px-4 py-3 text-right">
                {formatTokens(row.total_input_tokens)}
              </td>
              <td className="px-4 py-3 text-right">
                {formatTokens(row.total_output_tokens)}
              </td>
            </tr>
          ))}
          {data.length === 0 && (
            <tr>
              <td
                colSpan={5}
                className="px-4 py-8 text-center text-muted-foreground"
              >
                No cost data yet
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}

function LogsTable({
  logs,
  pagination,
  onPageChange,
}: {
  logs: CostLogRow[];
  pagination: Pagination | null;
  onPageChange: (page: number) => void;
}) {
  return (
    <div className="flex flex-col gap-4">
      <div className="overflow-x-auto">
        <table className="w-full text-left text-sm">
          <thead className="border-b text-xs uppercase text-muted-foreground">
            <tr>
              <th className="px-3 py-3">Time</th>
              <th className="px-3 py-3">User</th>
              <th className="px-3 py-3">Block</th>
              <th className="px-3 py-3">Provider</th>
              <th className="px-3 py-3">Model</th>
              <th className="px-3 py-3 text-right">Cost</th>
              <th className="px-3 py-3 text-right">Tokens (in/out)</th>
              <th className="px-3 py-3">Session</th>
            </tr>
          </thead>
          <tbody>
            {logs.map((log) => (
              <tr key={log.id} className="border-b hover:bg-muted">
                <td className="whitespace-nowrap px-3 py-2 text-xs">
                  {new Date(log.created_at).toLocaleString()}
                </td>
                <td className="px-3 py-2 text-xs">
                  {log.email || log.user_id.slice(0, 8)}
                </td>
                <td className="px-3 py-2 text-xs font-medium">
                  {log.block_name}
                </td>
                <td className="px-3 py-2 text-xs">{log.provider}</td>
                <td className="px-3 py-2 text-xs">{log.model || "-"}</td>
                <td className="px-3 py-2 text-right text-xs">
                  {log.cost_microdollars != null
                    ? formatMicrodollars(log.cost_microdollars)
                    : "-"}
                </td>
                <td className="px-3 py-2 text-right text-xs">
                  {log.input_tokens != null || log.output_tokens != null
                    ? `${formatTokens(log.input_tokens ?? 0)} / ${formatTokens(log.output_tokens ?? 0)}`
                    : "-"}
                </td>
                <td className="px-3 py-2 text-xs text-muted-foreground">
                  {log.graph_exec_id?.slice(0, 8) || "-"}
                </td>
              </tr>
            ))}
            {logs.length === 0 && (
              <tr>
                <td
                  colSpan={8}
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

export { PlatformCostContent };
