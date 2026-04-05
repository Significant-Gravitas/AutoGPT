"use client";

import { useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import type { PlatformCostDashboard } from "@/app/api/__generated__/models/platformCostDashboard";
import type { CostLogRow } from "@/app/api/__generated__/models/costLogRow";
import type { Pagination } from "@/app/api/__generated__/models/pagination";
import type { PlatformCostLogsResponse } from "@/app/api/__generated__/models/platformCostLogsResponse";
import { getPlatformCostDashboard, getPlatformCostLogs } from "../actions";
import { estimateCostForRow } from "../helpers";

interface InitialSearchParams {
  start?: string;
  end?: string;
  provider?: string;
  user_id?: string;
  page?: string;
  tab?: string;
}

// URL holds UTC ISO; datetime-local inputs need local "YYYY-MM-DDTHH:mm".
function toLocalInput(iso: string) {
  if (!iso) return "";
  const d = new Date(iso);
  if (isNaN(d.getTime())) return "";
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

// datetime-local emits naive local time; convert to UTC ISO so the
// backend filter window matches what the admin sees in their browser.
function toUtcIso(local: string) {
  if (!local) return "";
  const d = new Date(local);
  return isNaN(d.getTime()) ? "" : d.toISOString();
}

export function usePlatformCostContent(searchParams: InitialSearchParams) {
  const router = useRouter();
  const urlParams = useSearchParams();

  const [dashboard, setDashboard] = useState<PlatformCostDashboard | null>(
    null,
  );
  const [logs, setLogs] = useState<CostLogRow[]>([]);
  const [pagination, setPagination] = useState<Pagination | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  // Rate overrides keyed on `${provider}:${tracking_type}` so the same
  // provider can have independent rates per billing model.
  const [rateOverrides, setRateOverrides] = useState<Record<string, number>>(
    {},
  );

  const tab = urlParams.get("tab") || searchParams.tab || "overview";
  const page = parseInt(urlParams.get("page") || searchParams.page || "1", 10);
  const startDate = urlParams.get("start") || searchParams.start || "";
  const endDate = urlParams.get("end") || searchParams.end || "";
  const providerFilter =
    urlParams.get("provider") || searchParams.provider || "";
  const userFilter = urlParams.get("user_id") || searchParams.user_id || "";

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
    updateUrl({
      start: toUtcIso(startInput),
      end: toUtcIso(endInput),
      provider: providerInput,
      user_id: userInput,
      page: "1",
    });
  }

  function handleRateOverride(key: string, val: number) {
    setRateOverrides((prev) => ({ ...prev, [key]: val }));
  }

  const totalEstimatedCost =
    dashboard?.by_provider.reduce((sum, row) => {
      const est = estimateCostForRow(row, rateOverrides);
      return sum + (est ?? 0);
    }, 0) ?? 0;

  return {
    // Data
    dashboard,
    logs,
    pagination,
    loading,
    error,
    totalEstimatedCost,
    // URL state
    tab,
    page,
    // Filter inputs (uncommitted)
    startInput,
    setStartInput,
    endInput,
    setEndInput,
    providerInput,
    setProviderInput,
    userInput,
    setUserInput,
    // Rate overrides
    rateOverrides,
    handleRateOverride,
    // Actions
    updateUrl,
    handleFilter,
  };
}
