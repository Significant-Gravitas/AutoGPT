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

  const tab = searchParams.tab || "overview";
  const page = searchParams.page ? parseInt(searchParams.page) : 1;

  const [startDate, setStartDate] = useState(searchParams.start || "");
  const [endDate, setEndDate] = useState(searchParams.end || "");
  const [providerFilter, setProviderFilter] = useState(
    searchParams.provider || "",
  );
  const [userFilter, setUserFilter] = useState(searchParams.user_id || "");

  useEffect(() => {
    async function load() {
      setLoading(true);
      const filters: Record<string, string> = {};
      if (startDate) filters.start = startDate;
      if (endDate) filters.end = endDate;
      if (providerFilter) filters.provider = providerFilter;
      if (userFilter) filters.user_id = userFilter;

      const [dashData, logsData] = await Promise.all([
        getPlatformCostDashboard(filters),
        getPlatformCostLogs({ ...filters, page, page_size: 50 }),
      ]);
      setDashboard(dashData);
      setLogs(logsData.logs);
      setPagination(logsData.pagination);
      setLoading(false);
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
      start: startDate,
      end: endDate,
      provider: providerFilter,
      user_id: userFilter,
      page: "1",
    });
  }

  return (
    <div className="flex flex-col gap-6">
      {/* Filters */}
      <div className="flex flex-wrap items-end gap-3 rounded-lg border p-4">
        <div className="flex flex-col gap-1">
          <label className="text-sm text-gray-500">Start Date</label>
          <input
            type="datetime-local"
            className="rounded border px-3 py-1.5 text-sm"
            value={startDate}
            onChange={(e) => setStartDate(e.target.value)}
          />
        </div>
        <div className="flex flex-col gap-1">
          <label className="text-sm text-gray-500">End Date</label>
          <input
            type="datetime-local"
            className="rounded border px-3 py-1.5 text-sm"
            value={endDate}
            onChange={(e) => setEndDate(e.target.value)}
          />
        </div>
        <div className="flex flex-col gap-1">
          <label className="text-sm text-gray-500">Provider</label>
          <input
            type="text"
            placeholder="e.g. openai"
            className="rounded border px-3 py-1.5 text-sm"
            value={providerFilter}
            onChange={(e) => setProviderFilter(e.target.value)}
          />
        </div>
        <div className="flex flex-col gap-1">
          <label className="text-sm text-gray-500">User ID</label>
          <input
            type="text"
            placeholder="Filter by user"
            className="rounded border px-3 py-1.5 text-sm"
            value={userFilter}
            onChange={(e) => setUserFilter(e.target.value)}
          />
        </div>
        <button
          onClick={handleFilter}
          className="rounded bg-blue-600 px-4 py-1.5 text-sm text-white hover:bg-blue-700"
        >
          Apply
        </button>
      </div>

      {loading ? (
        <div className="py-10 text-center text-gray-500">Loading...</div>
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
                value={dashboard.by_user.length.toString()}
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
                    : "text-gray-500 hover:text-gray-700"
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
      <div className="text-sm text-gray-500">{label}</div>
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
        <thead className="border-b text-xs uppercase text-gray-500">
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
            <tr key={row.provider} className="border-b hover:bg-gray-50">
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
              <td colSpan={5} className="px-4 py-8 text-center text-gray-400">
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
        <thead className="border-b text-xs uppercase text-gray-500">
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
            <tr key={row.user_id} className="border-b hover:bg-gray-50">
              <td className="px-4 py-3">
                <div className="font-medium">{row.email || "Unknown"}</div>
                <div className="text-xs text-gray-400">{row.user_id}</div>
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
              <td colSpan={5} className="px-4 py-8 text-center text-gray-400">
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
          <thead className="border-b text-xs uppercase text-gray-500">
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
              <tr key={log.id} className="border-b hover:bg-gray-50">
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
                <td className="px-3 py-2 text-xs text-gray-400">
                  {log.graph_exec_id?.slice(0, 8) || "-"}
                </td>
              </tr>
            ))}
            {logs.length === 0 && (
              <tr>
                <td colSpan={8} className="px-4 py-8 text-center text-gray-400">
                  No logs found
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {pagination && pagination.total_pages > 1 && (
        <div className="flex items-center justify-between px-4">
          <span className="text-sm text-gray-500">
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
