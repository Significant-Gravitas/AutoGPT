"use client";

import { useEffect, useState } from "react";
import type { PlatformCostDashboard } from "@/app/api/__generated__/models/platformCostDashboard";
import type { CostLogRow } from "@/app/api/__generated__/models/costLogRow";
import type { Pagination } from "@/app/api/__generated__/models/pagination";
import type { PlatformCostLogsResponse } from "@/app/api/__generated__/models/platformCostLogsResponse";
import { getPlatformCostDashboard, getPlatformCostLogs } from "../actions";
import { estimateCostForRow, formatMicrodollars } from "../helpers";
import { useRouter, useSearchParams } from "next/navigation";
import { SummaryCard } from "./SummaryCard";
import { ProviderTable } from "./ProviderTable";
import { UserTable } from "./UserTable";
import { LogsTable } from "./LogsTable";

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
  const [costPerRunOverrides, setCostPerRunOverrides] = useState<
    Record<string, number>
  >({});

  const tab = urlParams.get("tab") || searchParams.tab || "overview";
  const page = parseInt(urlParams.get("page") || searchParams.page || "1", 10);
  const startDate = urlParams.get("start") || searchParams.start || "";
  const endDate = urlParams.get("end") || searchParams.end || "";
  const providerFilter =
    urlParams.get("provider") || searchParams.provider || "";
  const userFilter = urlParams.get("user_id") || searchParams.user_id || "";

  // URL holds UTC ISO; datetime-local inputs need local "YYYY-MM-DDTHH:mm".
  const toLocalInput = (iso: string) => {
    if (!iso) return "";
    const d = new Date(iso);
    if (isNaN(d.getTime())) return "";
    const pad = (n: number) => String(n).padStart(2, "0");
    return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
  };

  const [startInput, setStartInput] = useState(toLocalInput(startDate));
  const [endInput, setEndInput] = useState(toLocalInput(endDate));
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
        if (dashData) setDashboard(dashData);
        if (logsData) {
          setLogs((logsData as PlatformCostLogsResponse).logs || []);
          setPagination(
            (logsData as PlatformCostLogsResponse).pagination || null,
          );
        }
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
    // datetime-local emits naive local time; convert to UTC ISO so the
    // backend filter window matches what the admin sees in their browser.
    const toUtcIso = (local: string) => {
      if (!local) return "";
      const d = new Date(local);
      return isNaN(d.getTime()) ? "" : d.toISOString();
    };
    updateUrl({
      start: toUtcIso(startInput),
      end: toUtcIso(endInput),
      provider: providerInput,
      user_id: userInput,
      page: "1",
    });
  }

  const totalEstimatedCost =
    dashboard?.by_provider.reduce((sum, row) => {
      const est = estimateCostForRow(row, costPerRunOverrides);
      return sum + (est ?? 0);
    }, 0) ?? 0;

  return (
    <div className="flex flex-col gap-6">
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
          {dashboard && (
            <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
              <SummaryCard
                label="Known Cost"
                value={formatMicrodollars(dashboard.total_cost_microdollars)}
                subtitle="From providers that report USD cost"
              />
              <SummaryCard
                label="Estimated Total"
                value={formatMicrodollars(totalEstimatedCost)}
                subtitle="Including per-run cost estimates"
              />
              <SummaryCard
                label="Total Requests"
                value={dashboard.total_requests.toLocaleString()}
              />
              <SummaryCard
                label="Active Users"
                value={dashboard.total_users.toLocaleString()}
              />
            </div>
          )}

          <div className="flex gap-2 border-b">
            {["overview", "by-user", "logs"].map((t) => (
              <button
                key={t}
                onClick={() => updateUrl({ tab: t, page: "1" })}
                className={`px-4 py-2 text-sm font-medium ${tab === t ? "border-b-2 border-blue-600 text-blue-600" : "text-muted-foreground hover:text-foreground"}`}
              >
                {t === "overview"
                  ? "By Provider"
                  : t === "by-user"
                    ? "By User"
                    : "Raw Logs"}
              </button>
            ))}
          </div>

          {tab === "overview" && dashboard && (
            <ProviderTable
              data={dashboard.by_provider}
              costPerRunOverrides={costPerRunOverrides}
              onCostOverride={(provider, val) =>
                setCostPerRunOverrides((prev) => ({ ...prev, [provider]: val }))
              }
            />
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

export { PlatformCostContent };
