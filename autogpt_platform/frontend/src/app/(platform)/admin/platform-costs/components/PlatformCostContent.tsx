"use client";

import { Alert, AlertDescription } from "@/components/molecules/Alert/Alert";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { formatMicrodollars } from "../helpers";
import { SummaryCard } from "./SummaryCard";
import { ProviderTable } from "./ProviderTable";
import { UserTable } from "./UserTable";
import { LogsTable } from "./LogsTable";
import { usePlatformCostContent } from "./usePlatformCostContent";

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

export function PlatformCostContent({ searchParams }: Props) {
  const {
    dashboard,
    logs,
    pagination,
    loading,
    error,
    totalEstimatedCost,
    tab,
    startInput,
    setStartInput,
    endInput,
    setEndInput,
    providerInput,
    setProviderInput,
    userInput,
    setUserInput,
    rateOverrides,
    handleRateOverride,
    updateUrl,
    handleFilter,
  } = usePlatformCostContent(searchParams);

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-wrap items-end gap-3 rounded-lg border p-4">
        <div className="flex flex-col gap-1">
          <label htmlFor="start-date" className="text-sm text-muted-foreground">
            Start Date{" "}
            <span className="text-xs">
              (local time — defaults to last 30 days)
            </span>
          </label>
          <input
            id="start-date"
            type="datetime-local"
            className="rounded border px-3 py-1.5 text-sm"
            value={startInput}
            onChange={(e) => setStartInput(e.target.value)}
          />
        </div>
        <div className="flex flex-col gap-1">
          <label htmlFor="end-date" className="text-sm text-muted-foreground">
            End Date <span className="text-xs">(local time)</span>
          </label>
          <input
            id="end-date"
            type="datetime-local"
            className="rounded border px-3 py-1.5 text-sm"
            value={endInput}
            onChange={(e) => setEndInput(e.target.value)}
          />
        </div>
        <div className="flex flex-col gap-1">
          <label
            htmlFor="provider-filter"
            className="text-sm text-muted-foreground"
          >
            Provider
          </label>
          <input
            id="provider-filter"
            type="text"
            placeholder="e.g. openai"
            className="rounded border px-3 py-1.5 text-sm"
            value={providerInput}
            onChange={(e) => setProviderInput(e.target.value)}
          />
        </div>
        <div className="flex flex-col gap-1">
          <label
            htmlFor="user-id-filter"
            className="text-sm text-muted-foreground"
          >
            User ID
          </label>
          <input
            id="user-id-filter"
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
        <button
          onClick={() => {
            setStartInput("");
            setEndInput("");
            setProviderInput("");
            setUserInput("");
            updateUrl({
              start: "",
              end: "",
              provider: "",
              user_id: "",
              page: "1",
            });
          }}
          className="rounded border px-4 py-1.5 text-sm hover:bg-muted"
        >
          Clear
        </button>
      </div>

      {error && (
        <Alert variant="error">
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      )}

      {loading ? (
        <div className="flex flex-col gap-4">
          <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
            {[...Array(4)].map((_, i) => (
              <Skeleton key={i} className="h-20 rounded-lg" />
            ))}
          </div>
          <Skeleton className="h-8 w-48 rounded" />
          <Skeleton className="h-64 rounded-lg" />
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

          <div
            role="tablist"
            aria-label="Cost view tabs"
            className="flex gap-2 border-b"
          >
            {["overview", "by-user", "logs"].map((t) => (
              <button
                key={t}
                id={`tab-${t}`}
                role="tab"
                aria-selected={tab === t}
                aria-controls={`tabpanel-${t}`}
                onClick={() => updateUrl({ tab: t, page: "1" })}
                className={`px-4 py-2 text-sm font-medium ${tab === t ? "border-b-2 border-primary text-primary" : "text-muted-foreground hover:text-foreground"}`}
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
            <div
              role="tabpanel"
              id="tabpanel-overview"
              aria-labelledby="tab-overview"
            >
              <ProviderTable
                data={dashboard.by_provider}
                rateOverrides={rateOverrides}
                onRateOverride={handleRateOverride}
              />
            </div>
          )}
          {tab === "by-user" && dashboard && (
            <div
              role="tabpanel"
              id="tabpanel-by-user"
              aria-labelledby="tab-by-user"
            >
              <UserTable data={dashboard.by_user} />
            </div>
          )}
          {tab === "logs" && (
            <div role="tabpanel" id="tabpanel-logs" aria-labelledby="tab-logs">
              <LogsTable
                logs={logs}
                pagination={pagination}
                onPageChange={(p) => updateUrl({ page: p.toString() })}
              />
            </div>
          )}
        </>
      )}
    </div>
  );
}
