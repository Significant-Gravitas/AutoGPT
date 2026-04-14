"use client";

import { Alert, AlertDescription } from "@/components/molecules/Alert/Alert";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { formatMicrodollars, formatTokens } from "../helpers";
import { SummaryCard } from "./SummaryCard";
import { ProviderTable } from "./ProviderTable";
import { UserTable } from "./UserTable";
import { LogsTable } from "./LogsTable";
import { usePlatformCostContent } from "./usePlatformCostContent";
import type { CostBucket } from "@/app/api/__generated__/models/costBucket";

interface Props {
  searchParams: {
    start?: string;
    end?: string;
    provider?: string;
    user_id?: string;
    model?: string;
    block_name?: string;
    tracking_type?: string;
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
  } = usePlatformCostContent(searchParams);

  const summaryCards: { label: string; value: string; subtitle?: string }[] =
    dashboard
      ? [
          {
            label: "Known Cost",
            value: formatMicrodollars(dashboard.total_cost_microdollars),
            subtitle: "From providers that report USD cost",
          },
          {
            label: "Estimated Total",
            value: formatMicrodollars(totalEstimatedCost),
            subtitle: "Including per-run cost estimates",
          },
          {
            label: "Total Requests",
            value: dashboard.total_requests.toLocaleString(),
          },
          {
            label: "Active Users",
            value: dashboard.total_users.toLocaleString(),
          },
          {
            label: "Avg Cost / Request",
            value: formatMicrodollars(
              dashboard.avg_cost_microdollars_per_request ?? 0,
            ),
            subtitle: "Known cost divided by cost-bearing requests",
          },
          {
            label: "Avg Input Tokens",
            value: Math.round(
              dashboard.avg_input_tokens_per_request ?? 0,
            ).toLocaleString(),
            subtitle: "Prompt tokens per request (context size)",
          },
          {
            label: "Avg Output Tokens",
            value: Math.round(
              dashboard.avg_output_tokens_per_request ?? 0,
            ).toLocaleString(),
            subtitle: "Completion tokens per request (response length)",
          },
          {
            label: "Total Tokens",
            value: `${formatTokens(dashboard.total_input_tokens ?? 0)} in / ${formatTokens(dashboard.total_output_tokens ?? 0)} out`,
            subtitle: "Prompt vs completion token split",
          },
          {
            label: "Typical Cost (P50)",
            value: formatMicrodollars(dashboard.cost_p50_microdollars ?? 0),
            subtitle: "Median cost per request",
          },
          {
            label: "Upper Cost (P75)",
            value: formatMicrodollars(dashboard.cost_p75_microdollars ?? 0),
            subtitle: "75th percentile cost",
          },
          {
            label: "High Cost (P95)",
            value: formatMicrodollars(dashboard.cost_p95_microdollars ?? 0),
            subtitle: "95th percentile cost",
          },
          {
            label: "Peak Cost (P99)",
            value: formatMicrodollars(dashboard.cost_p99_microdollars ?? 0),
            subtitle: "99th percentile cost",
          },
        ]
      : [];

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
        <div className="flex flex-col gap-1">
          <label
            htmlFor="model-filter"
            className="text-sm text-muted-foreground"
          >
            Model
          </label>
          <input
            id="model-filter"
            type="text"
            placeholder="e.g. gpt-4o"
            className="rounded border px-3 py-1.5 text-sm"
            value={modelInput}
            onChange={(e) => setModelInput(e.target.value)}
          />
        </div>
        <div className="flex flex-col gap-1">
          <label
            htmlFor="block-filter"
            className="text-sm text-muted-foreground"
          >
            Block
          </label>
          <input
            id="block-filter"
            type="text"
            placeholder="e.g. LLMBlock"
            className="rounded border px-3 py-1.5 text-sm"
            value={blockInput}
            onChange={(e) => setBlockInput(e.target.value)}
          />
        </div>
        <div className="flex flex-col gap-1">
          <label
            htmlFor="type-filter"
            className="text-sm text-muted-foreground"
          >
            Type
          </label>
          <input
            id="type-filter"
            type="text"
            placeholder="e.g. tokens"
            className="rounded border px-3 py-1.5 text-sm"
            value={typeInput}
            onChange={(e) => setTypeInput(e.target.value)}
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
            setModelInput("");
            setBlockInput("");
            setTypeInput("");
            updateUrl({
              start: "",
              end: "",
              provider: "",
              user_id: "",
              model: "",
              block_name: "",
              tracking_type: "",
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
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 md:grid-cols-4">
            {/* 12 skeleton placeholders — one per summary card */}
            {Array.from({ length: 12 }, (_, i) => (
              <Skeleton key={i} className="h-20 rounded-lg" />
            ))}
          </div>
          <Skeleton className="h-32 rounded-lg" />
          <Skeleton className="h-8 w-48 rounded" />
          <Skeleton className="h-64 rounded-lg" />
        </div>
      ) : (
        <>
          {dashboard && (
            <>
              <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 md:grid-cols-4">
                {summaryCards.map((card) => (
                  <SummaryCard
                    key={card.label}
                    label={card.label}
                    value={card.value}
                    subtitle={card.subtitle}
                  />
                ))}
              </div>

              {dashboard.cost_buckets && dashboard.cost_buckets.length > 0 && (
                <div className="rounded-lg border p-4">
                  <h3 className="mb-3 text-sm font-medium">
                    Cost Distribution by Bucket
                  </h3>
                  <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 md:grid-cols-6">
                    {dashboard.cost_buckets.map((b: CostBucket) => (
                      <div
                        key={b.bucket}
                        className="flex flex-col items-center rounded border p-2 text-center"
                      >
                        <span className="text-xs text-muted-foreground">
                          {b.bucket}
                        </span>
                        <span className="text-lg font-semibold">
                          {b.count.toLocaleString()}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </>
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
                onExport={handleExport}
                exporting={exporting}
              />
            </div>
          )}
        </>
      )}
    </div>
  );
}
