"use client";

import { useRouter, useSearchParams } from "next/navigation";
import { useState } from "react";
import {
  getV2ExportPlatformCostLogs,
  useGetV2GetPlatformCostDashboard,
  useGetV2GetPlatformCostLogs,
} from "@/app/api/__generated__/endpoints/admin/admin";
import { okData } from "@/app/api/helpers";
import {
  buildCostLogsCsv,
  estimateCostForRow,
  toLocalInput,
  toUtcIso,
} from "../helpers";

interface InitialSearchParams {
  start?: string;
  end?: string;
  provider?: string;
  user_id?: string;
  model?: string;
  block_name?: string;
  tracking_type?: string;
  page?: string;
  tab?: string;
}

export function usePlatformCostContent(searchParams: InitialSearchParams) {
  const router = useRouter();
  const urlParams = useSearchParams();

  const tab = urlParams.get("tab") || searchParams.tab || "overview";
  const page = parseInt(urlParams.get("page") || searchParams.page || "1", 10);
  const startDate = urlParams.get("start") || searchParams.start || "";
  const endDate = urlParams.get("end") || searchParams.end || "";
  const providerFilter =
    urlParams.get("provider") || searchParams.provider || "";
  const userFilter = urlParams.get("user_id") || searchParams.user_id || "";
  const modelFilter = urlParams.get("model") || searchParams.model || "";
  const blockFilter =
    urlParams.get("block_name") || searchParams.block_name || "";
  const typeFilter =
    urlParams.get("tracking_type") || searchParams.tracking_type || "";

  const [startInput, setStartInput] = useState(toLocalInput(startDate));
  const [endInput, setEndInput] = useState(toLocalInput(endDate));
  const [providerInput, setProviderInput] = useState(providerFilter);
  const [userInput, setUserInput] = useState(userFilter);
  const [modelInput, setModelInput] = useState(modelFilter);
  const [blockInput, setBlockInput] = useState(blockFilter);
  const [typeInput, setTypeInput] = useState(typeFilter);
  const [rateOverrides, setRateOverrides] = useState<Record<string, number>>(
    {},
  );
  const [exporting, setExporting] = useState(false);

  // Pass ISO date strings through `as unknown as Date` so Orval's URL builder
  // forwards them as-is. Date.toString() produces a format FastAPI rejects;
  // strings pass through .toString() unchanged.
  const filterParams = {
    start: (startDate || undefined) as unknown as Date | undefined,
    end: (endDate || undefined) as unknown as Date | undefined,
    provider: providerFilter || undefined,
    user_id: userFilter || undefined,
    model: modelFilter || undefined,
    block_name: blockFilter || undefined,
    tracking_type: typeFilter || undefined,
  };

  const {
    data: dashboard,
    isLoading: dashLoading,
    error: dashError,
  } = useGetV2GetPlatformCostDashboard(filterParams, {
    query: { select: okData },
  });

  const {
    data: logsResponse,
    isLoading: logsLoading,
    error: logsError,
  } = useGetV2GetPlatformCostLogs(
    { ...filterParams, page, page_size: 50 },
    { query: { select: okData } },
  );

  const loading = dashLoading || logsLoading;
  const error = dashError
    ? dashError instanceof Error
      ? dashError.message
      : "Failed to load dashboard"
    : logsError
      ? logsError instanceof Error
        ? logsError.message
        : "Failed to load logs"
      : null;

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
      model: modelInput,
      block_name: blockInput,
      tracking_type: typeInput,
      page: "1",
    });
  }

  function handleRateOverride(key: string, val: number | null) {
    setRateOverrides((prev) => {
      if (val === null) {
        const { [key]: _, ...rest } = prev;
        return rest;
      }
      return { ...prev, [key]: val };
    });
  }

  async function handleExport() {
    setExporting(true);
    try {
      const response = await getV2ExportPlatformCostLogs(filterParams);
      const data = okData(response);
      if (!data) throw new Error("Export failed: unexpected response");
      const csv = buildCostLogsCsv(data.logs);
      const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `platform_costs_${new Date().toISOString().slice(0, 10)}.csv`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      if (data.truncated) {
        // eslint-disable-next-line no-console
        console.warn(
          `Export truncated: only the first ${data.total_rows} rows were included.`,
        );
      }
    } finally {
      setExporting(false);
    }
  }

  const totalEstimatedCost =
    dashboard?.by_provider.reduce((sum, row) => {
      const est = estimateCostForRow(row, rateOverrides);
      return sum + (est ?? 0);
    }, 0) ?? 0;

  return {
    dashboard: dashboard ?? null,
    logs: logsResponse?.logs ?? [],
    pagination: logsResponse?.pagination ?? null,
    loading,
    error,
    totalEstimatedCost,
    tab,
    page,
    startInput,
    setStartInput,
    endInput,
    setEndInput,
    providerInput,
    setProviderInput,
    userInput,
    setUserInput,
    modelInput,
    setModelInput,
    blockInput,
    setBlockInput,
    typeInput,
    setTypeInput,
    rateOverrides,
    handleRateOverride,
    updateUrl,
    handleFilter,
    exporting,
    handleExport,
  };
}
