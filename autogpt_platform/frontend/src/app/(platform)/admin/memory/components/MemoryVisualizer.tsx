"use client";

import { useMemoryVisualizer } from "./useMemoryVisualizer";

type StatusFilter = "any" | "active" | "superseded" | "contradicted";

const STATUS_OPTIONS: StatusFilter[] = [
  "any",
  "active",
  "superseded",
  "contradicted",
];

export function MemoryVisualizer() {
  const {
    overview,
    entities,
    facts,
    communities,
    rebuild,
    triggerRebuild,
    statusFilter,
    setStatusFilter,
    force,
    setForce,
    overviewData,
    entitiesData,
    factsData,
    communitiesData,
    rebuildData,
  } = useMemoryVisualizer();

  const entityItems = entitiesData?.items ?? [];
  const factItems = factsData?.items ?? [];
  const communityItems = communitiesData?.items ?? [];

  return (
    <div className="space-y-6">
      <OverviewStrip
        loading={overview.isLoading}
        error={overview.error}
        data={overviewData}
      />

      <div className="rounded-md border bg-white p-4">
        <div className="flex flex-wrap items-center gap-3">
          <button
            type="button"
            onClick={triggerRebuild}
            disabled={rebuild.isPending}
            className="rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50"
          >
            {rebuild.isPending ? "Rebuilding…" : "Rebuild communities now"}
          </button>
          <label className="flex items-center gap-2 text-sm text-gray-700">
            <input
              type="checkbox"
              checked={force}
              onChange={(e) => setForce(e.target.checked)}
            />
            Force (bypass activity gate)
          </label>
          {rebuildData && (
            <span className="text-xs text-gray-500">
              Last:{" "}
              {rebuildData.skipped
                ? `skipped (${rebuildData.skip_reason})`
                : `${rebuildData.elapsed_seconds?.toFixed(1)}s`}
            </span>
          )}
        </div>
      </div>

      <Section title={`Entities (${entityItems.length || "…"})`}>
        <Table
          loading={entities.isLoading}
          error={entities.error}
          empty="No entities yet — start a chat session to populate memory."
          rows={entityItems}
          columns={[
            { label: "Name", get: (e) => e.name },
            { label: "Summary", get: (e) => e.summary ?? "—" },
            { label: "UUID", get: (e) => e.uuid, className: "font-mono text-xs" },
          ]}
        />
      </Section>

      <Section
        title={`Facts (${factItems.length || "…"})`}
        controls={
          <select
            value={statusFilter}
            onChange={(e) => setStatusFilter(e.target.value as StatusFilter)}
            className="rounded-md border px-2 py-1 text-sm"
          >
            {STATUS_OPTIONS.map((s) => (
              <option key={s} value={s}>
                status: {s}
              </option>
            ))}
          </select>
        }
      >
        <Table
          loading={facts.isLoading}
          error={facts.error}
          empty="No facts at this filter."
          rows={factItems}
          columns={[
            { label: "Source", get: (f) => f.source },
            { label: "Relation", get: (f) => f.name ?? "—" },
            { label: "Target", get: (f) => f.target },
            { label: "Status", get: (f) => f.status ?? "—" },
            { label: "Scope", get: (f) => f.scope ?? "—" },
            {
              label: "Confidence",
              get: (f) => f.confidence?.toFixed(2) ?? "—",
            },
          ]}
        />
      </Section>

      <Section title={`Communities (${communityItems.length || "…"})`}>
        <Table
          loading={communities.isLoading}
          error={communities.error}
          empty="No communities yet — click 'Rebuild communities now' above."
          rows={communityItems}
          columns={[
            { label: "Name", get: (c) => c.name ?? "—" },
            { label: "Members", get: (c) => String(c.member_count) },
            {
              label: "Summary",
              get: (c) => c.summary ?? "—",
              className: "max-w-md truncate",
            },
          ]}
        />
      </Section>
    </div>
  );
}

interface OverviewStripProps {
  loading: boolean;
  error: unknown;
  data:
    | {
        entities: number;
        episodes: number;
        relates_to_edges: number;
        mentions_edges: number;
        communities: number;
      }
    | undefined;
}

function OverviewStrip({ loading, error, data }: OverviewStripProps) {
  if (loading) {
    return (
      <div className="rounded-md border bg-white p-4 text-sm text-gray-500">
        Loading overview…
      </div>
    );
  }
  if (error) {
    return (
      <div className="rounded-md border border-red-200 bg-red-50 p-4 text-sm text-red-700">
        Failed to load overview: {String(error)}
      </div>
    );
  }
  if (!data) return null;
  return (
    <div className="grid grid-cols-2 gap-3 md:grid-cols-5">
      {[
        { label: "Entities", value: data.entities },
        { label: "Episodes", value: data.episodes },
        { label: "Facts (edges)", value: data.relates_to_edges },
        { label: "Mentions", value: data.mentions_edges },
        { label: "Communities", value: data.communities },
      ].map((stat) => (
        <div
          key={stat.label}
          className="rounded-md border bg-white p-4 text-center"
        >
          <div className="text-2xl font-bold">{stat.value}</div>
          <div className="text-xs uppercase tracking-wide text-gray-500">
            {stat.label}
          </div>
        </div>
      ))}
    </div>
  );
}

interface SectionProps {
  title: string;
  controls?: React.ReactNode;
  children: React.ReactNode;
}

function Section({ title, controls, children }: SectionProps) {
  return (
    <div className="rounded-md border bg-white">
      <div className="flex items-center justify-between border-b px-4 py-3">
        <h2 className="text-lg font-semibold">{title}</h2>
        {controls}
      </div>
      <div className="overflow-x-auto p-4">{children}</div>
    </div>
  );
}

interface Column<T> {
  label: string;
  get: (row: T) => string;
  className?: string;
}

interface TableProps<T extends { uuid: string }> {
  loading: boolean;
  error: unknown;
  empty: string;
  rows: T[];
  columns: Column<T>[];
}

function Table<T extends { uuid: string }>({
  loading,
  error,
  empty,
  rows,
  columns,
}: TableProps<T>) {
  if (loading) return <div className="text-sm text-gray-500">Loading…</div>;
  if (error)
    return (
      <div className="text-sm text-red-700">Failed to load: {String(error)}</div>
    );
  if (rows.length === 0)
    return <div className="text-sm text-gray-500">{empty}</div>;
  return (
    <table className="w-full text-sm">
      <thead>
        <tr className="border-b text-left">
          {columns.map((c) => (
            <th key={c.label} className="px-2 py-2 font-medium text-gray-700">
              {c.label}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => (
          <tr key={row.uuid} className="border-b last:border-b-0">
            {columns.map((c) => (
              <td
                key={c.label}
                className={`px-2 py-2 align-top ${c.className ?? ""}`}
              >
                {c.get(row)}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}
