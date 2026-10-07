"use client";

import { Alert, AlertDescription } from "@/components/molecules/Alert/Alert";
import { Skeleton } from "@/components/atoms/Skeleton/Skeleton";
import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { formatMicrodollars, formatTokens } from "../helpers";
import { SummaryCard } from "./SummaryCard";
import { ProviderTable } from "./ProviderTable";
import { UserTable } from "./UserTable";
import { LogsTable } from "./LogsTable";
import { usePlatformCostContent } from "./usePlatformCostContent";
import type { CostBucket } from "@/app/api/__generated__/models/costBucket";
import { Text } from "@/components/atoms/Text/Text";

interface Props {
  searchParams: {
    start?: string;
    end?: string;
    provider?: string;
    user_id?: string;
    model?: string;
    block_name?: string;
    tracking_type?: string;
    graph_exec_id?: string;
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
    executionIDInput,
    setExecutionIDInput,
    executionPathInput,
    setExecutionPathInput,
    sourceInput,
    setSourceInput,
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
          <Input
            id="start-date"
            label="Start Date"
            labelVariant="body"
            labelClassName="text-muted-foreground"
            hint="(local time — defaults to last 30 days)"
            size="md"
            wrapperClassName="mb-0"
            type="datetime-local"
            value={startInput}
            onChange={(e) => setStartInput(e.target.value)}
          />
        </div>
        <div className="flex flex-col gap-1">
          <Input
            id="end-date"
            label="End Date"
            labelVariant="body"
            labelClassName="text-muted-foreground"
            hint="(local time)"
            size="md"
            wrapperClassName="mb-0"
            type="datetime-local"
            value={endInput}
            onChange={(e) => setEndInput(e.target.value)}
          />
        </div>
        <div className="flex flex-col gap-1">
          <Input
            id="provider-filter"
            label="Provider"
            labelVariant="body"
            labelClassName="text-muted-foreground"
            size="md"
            wrapperClassName="mb-0"
            type="text"
            placeholder="e.g. openai"
            value={providerInput}
            onChange={(e) => setProviderInput(e.target.value)}
          />
        </div>
        <div className="flex flex-col gap-1">
          <Input
            id="user-id-filter"
            label="User ID"
            labelVariant="body"
            labelClassName="text-muted-foreground"
            size="md"
            wrapperClassName="mb-0"
            type="text"
            placeholder="Filter by user"
            value={userInput}
            onChange={(e) => setUserInput(e.target.value)}
          />
        </div>
        <div className="flex flex-col gap-1">
          <Input
            id="model-filter"
            label="Model"
            labelVariant="body"
            labelClassName="text-muted-foreground"
            size="md"
            wrapperClassName="mb-0"
            type="text"
            placeholder="e.g. gpt-4o"
            value={modelInput}
            onChange={(e) => setModelInput(e.target.value)}
          />
        </div>
        <div className="flex flex-col gap-1">
          <Input
            id="block-filter"
            label="Block"
            labelVariant="body"
            labelClassName="text-muted-foreground"
            size="md"
            wrapperClassName="mb-0"
            type="text"
            placeholder="e.g. LLMBlock"
            value={blockInput}
            onChange={(e) => setBlockInput(e.target.value)}
          />
        </div>
        <div className="flex flex-col gap-1">
          <Input
            id="type-filter"
            label="Type"
            labelVariant="body"
            labelClassName="text-muted-foreground"
            size="md"
            wrapperClassName="mb-0"
            type="text"
            placeholder="e.g. tokens"
            value={typeInput}
            onChange={(e) => setTypeInput(e.target.value)}
          />
        </div>
        <div className="flex flex-col gap-1">
          <Input
            id="execution-id-filter"
            label="Execution ID"
            labelVariant="body"
            labelClassName="text-muted-foreground"
            size="md"
            wrapperClassName="mb-0"
            type="text"
            placeholder="Filter by execution"
            value={executionIDInput}
            onChange={(e) => setExecutionIDInput(e.target.value)}
          />
        </div>
        <div className="flex flex-col gap-1">
          <label
            htmlFor="execution-path-filter"
            className="text-sm text-muted-foreground"
          >
            Path
          </label>
          <select
            id="execution-path-filter"
            className="rounded-sm border px-3 py-1.5 text-sm"
            value={executionPathInput}
            onChange={(e) => setExecutionPathInput(e.target.value)}
          >
            <option value="">All</option>
            <option value="sync">sync</option>
            <option value="anthropic_batch">anthropic_batch</option>
            <option value="openai_batch">openai_batch</option>
            <option value="flex">flex</option>
            <option value="sync_baseline">sync_baseline (dream)</option>
          </select>
        </div>
        <div className="flex flex-col gap-1">
          <label
            htmlFor="source-filter"
            className="text-sm text-muted-foreground"
          >
            Source
          </label>
          <select
            id="source-filter"
            className="rounded-sm border px-3 py-1.5 text-sm"
            value={sourceInput}
            onChange={(e) => setSourceInput(e.target.value)}
          >
            <option value="">All</option>
            <option value="copilot">copilot</option>
            <option value="dream_pass">dream_pass</option>
          </select>
        </div>
        <Button variant="primary" size="md" onClick={handleFilter}>
          Apply
        </Button>
        <Button
          variant="secondary"
          size="md"
          onClick={() => {
            setStartInput("");
            setEndInput("");
            setProviderInput("");
            setUserInput("");
            setModelInput("");
            setBlockInput("");
            setTypeInput("");
            setExecutionIDInput("");
            setExecutionPathInput("");
            setSourceInput("");
            updateUrl({
              start: "",
              end: "",
              provider: "",
              user_id: "",
              model: "",
              block_name: "",
              tracking_type: "",
              graph_exec_id: "",
              execution_path: "",
              source: "",
              page: "1",
            });
          }}
        >
          Clear
        </Button>
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
          <Skeleton className="h-8 w-48 rounded-sm" />
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
                  <Text variant="body-medium" as="h3" className="mb-3">
                    Cost Distribution by Bucket
                  </Text>
                  <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 md:grid-cols-6">
                    {dashboard.cost_buckets.map((b: CostBucket) => (
                      <div
                        key={b.bucket}
                        className="flex flex-col items-center rounded-sm border p-2 text-center"
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
