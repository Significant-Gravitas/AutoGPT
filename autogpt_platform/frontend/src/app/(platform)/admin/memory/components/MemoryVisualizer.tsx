"use client";

import { useMemo, useState } from "react";
import { GraphCanvas } from "./GraphCanvas";
import { useMemoryVisualizer } from "./useMemoryVisualizer";

export function MemoryVisualizer() {
  const {
    overview,
    overviewData,
    graph,
    graphData,
    rebuild,
    triggerRebuild,
    rebuildData,
    force,
    setForce,
    includeEpisodes,
    setIncludeEpisodes,
    includeCommunities,
    setIncludeCommunities,
  } = useMemoryVisualizer();

  // Per-label / per-relationship visibility toggles. Selected via a Set
  // of "hidden" entries — empty set = show everything.
  const [hiddenNodeTypes, setHiddenNodeTypes] = useState<Set<string>>(
    new Set(),
  );
  const [hiddenEdgeTypes, setHiddenEdgeTypes] = useState<Set<string>>(
    new Set(),
  );
  const [selectedUuid, setSelectedUuid] = useState<string | null>(null);

  const nodes = graphData?.nodes ?? [];
  const edges = graphData?.edges ?? [];
  const truncated = graphData?.truncated ?? false;

  // Aggregate label / relationship counts for the sidebar pills.
  const { nodeTypeCounts, edgeTypeCounts } = useMemo(() => {
    const ntc = new Map<string, number>();
    const etc = new Map<string, number>();
    for (const n of nodes) {
      const k = n.type ?? n.label;
      ntc.set(k, (ntc.get(k) ?? 0) + 1);
    }
    for (const e of edges) {
      etc.set(e.label, (etc.get(e.label) ?? 0) + 1);
    }
    return { nodeTypeCounts: ntc, edgeTypeCounts: etc };
  }, [nodes, edges]);

  const selectedNode = useMemo(
    () => (selectedUuid ? nodes.find((n) => n.uuid === selectedUuid) : null),
    [selectedUuid, nodes],
  );
  const selectedNeighbors = useMemo(() => {
    if (!selectedUuid) return [] as { edge: (typeof edges)[number]; otherUuid: string }[];
    const result: { edge: (typeof edges)[number]; otherUuid: string }[] = [];
    for (const e of edges) {
      if (e.source === selectedUuid)
        result.push({ edge: e, otherUuid: e.target });
      else if (e.target === selectedUuid)
        result.push({ edge: e, otherUuid: e.source });
    }
    return result;
  }, [selectedUuid, edges]);

  return (
    <div className="space-y-4">
      <OverviewStrip
        loading={overview.isLoading}
        error={overview.error}
        data={overviewData}
      />

      <ControlBar
        onRebuild={triggerRebuild}
        rebuildPending={rebuild.isPending}
        rebuildResult={rebuildData}
        force={force}
        setForce={setForce}
        includeEpisodes={includeEpisodes}
        setIncludeEpisodes={setIncludeEpisodes}
        includeCommunities={includeCommunities}
        setIncludeCommunities={setIncludeCommunities}
        truncated={truncated}
        nodeCount={nodes.length}
        edgeCount={edges.length}
      />

      <div className="grid grid-cols-12 gap-4">
        <Sidebar
          nodeTypeCounts={nodeTypeCounts}
          edgeTypeCounts={edgeTypeCounts}
          hiddenNodeTypes={hiddenNodeTypes}
          setHiddenNodeTypes={setHiddenNodeTypes}
          hiddenEdgeTypes={hiddenEdgeTypes}
          setHiddenEdgeTypes={setHiddenEdgeTypes}
        />
        <div className="col-span-12 rounded-md border bg-white md:col-span-7">
          {graph.isLoading ? (
            <div className="flex h-[70vh] items-center justify-center text-sm text-gray-500">
              Loading graph…
            </div>
          ) : graph.error ? (
            <div className="p-6 text-sm text-red-700">
              Failed to load graph: {String(graph.error)}
            </div>
          ) : nodes.length === 0 ? (
            <div className="flex h-[70vh] flex-col items-center justify-center text-sm text-gray-500">
              No memory yet. Start a chat session to populate it.
            </div>
          ) : (
            <GraphCanvas
              nodes={nodes}
              edges={edges}
              hiddenNodeTypes={hiddenNodeTypes}
              hiddenEdgeTypes={hiddenEdgeTypes}
              selectedUuid={selectedUuid}
              onSelect={setSelectedUuid}
            />
          )}
        </div>
        <DetailPanel
          node={selectedNode ?? null}
          neighbors={selectedNeighbors}
          onClear={() => setSelectedUuid(null)}
        />
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

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
      <div className="rounded-md border bg-white p-3 text-sm text-gray-500">
        Loading overview…
      </div>
    );
  }
  if (error) {
    return (
      <div className="rounded-md border border-red-200 bg-red-50 p-3 text-sm text-red-700">
        Failed to load overview: {String(error)}
      </div>
    );
  }
  if (!data) return null;
  return (
    <div className="grid grid-cols-2 gap-2 md:grid-cols-5">
      {[
        { label: "Entities", value: data.entities },
        { label: "Episodes", value: data.episodes },
        { label: "Facts", value: data.relates_to_edges },
        { label: "Mentions", value: data.mentions_edges },
        { label: "Communities", value: data.communities },
      ].map((stat) => (
        <div
          key={stat.label}
          className="rounded-md border bg-white p-3 text-center"
        >
          <div className="text-xl font-bold">{stat.value}</div>
          <div className="text-xs uppercase tracking-wide text-gray-500">
            {stat.label}
          </div>
        </div>
      ))}
    </div>
  );
}

interface ControlBarProps {
  onRebuild: () => void;
  rebuildPending: boolean;
  rebuildResult: { skipped: boolean; skip_reason?: string | null; elapsed_seconds?: number | null } | undefined;
  force: boolean;
  setForce: (v: boolean) => void;
  includeEpisodes: boolean;
  setIncludeEpisodes: (v: boolean) => void;
  includeCommunities: boolean;
  setIncludeCommunities: (v: boolean) => void;
  truncated: boolean;
  nodeCount: number;
  edgeCount: number;
}

function ControlBar({
  onRebuild,
  rebuildPending,
  rebuildResult,
  force,
  setForce,
  includeEpisodes,
  setIncludeEpisodes,
  includeCommunities,
  setIncludeCommunities,
  truncated,
  nodeCount,
  edgeCount,
}: ControlBarProps) {
  return (
    <div className="flex flex-wrap items-center gap-3 rounded-md border bg-white p-3 text-sm">
      <button
        type="button"
        onClick={onRebuild}
        disabled={rebuildPending}
        className="rounded-md bg-blue-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-blue-700 disabled:opacity-50"
      >
        {rebuildPending ? "Rebuilding…" : "Rebuild communities"}
      </button>
      <label className="flex items-center gap-2 text-gray-700">
        <input
          type="checkbox"
          checked={force}
          onChange={(e) => setForce(e.target.checked)}
        />
        Force
      </label>
      <span className="mx-2 h-5 border-l border-gray-200" />
      <label className="flex items-center gap-2 text-gray-700">
        <input
          type="checkbox"
          checked={includeCommunities}
          onChange={(e) => setIncludeCommunities(e.target.checked)}
        />
        Communities
      </label>
      <label className="flex items-center gap-2 text-gray-700">
        <input
          type="checkbox"
          checked={includeEpisodes}
          onChange={(e) => setIncludeEpisodes(e.target.checked)}
        />
        Episodes (noisy)
      </label>
      <span className="ml-auto text-xs text-gray-500">
        {nodeCount} nodes · {edgeCount} edges
        {truncated && (
          <span className="ml-2 rounded bg-amber-100 px-2 py-0.5 text-amber-800">
            truncated
          </span>
        )}
        {rebuildResult && (
          <span className="ml-2">
            last rebuild:{" "}
            {rebuildResult.skipped
              ? `skipped (${rebuildResult.skip_reason})`
              : `${rebuildResult.elapsed_seconds?.toFixed(1)}s`}
          </span>
        )}
      </span>
    </div>
  );
}

interface SidebarProps {
  nodeTypeCounts: Map<string, number>;
  edgeTypeCounts: Map<string, number>;
  hiddenNodeTypes: Set<string>;
  setHiddenNodeTypes: (s: Set<string>) => void;
  hiddenEdgeTypes: Set<string>;
  setHiddenEdgeTypes: (s: Set<string>) => void;
}

function Sidebar({
  nodeTypeCounts,
  edgeTypeCounts,
  hiddenNodeTypes,
  setHiddenNodeTypes,
  hiddenEdgeTypes,
  setHiddenEdgeTypes,
}: SidebarProps) {
  function togglePill(set: Set<string>, key: string): Set<string> {
    const next = new Set(set);
    if (next.has(key)) next.delete(key);
    else next.add(key);
    return next;
  }
  return (
    <div className="col-span-12 space-y-3 md:col-span-3">
      <div className="rounded-md border bg-white p-3">
        <h3 className="mb-2 text-xs font-semibold uppercase text-gray-500">
          Labels
        </h3>
        <div className="flex flex-wrap gap-1.5">
          {[...nodeTypeCounts.entries()]
            .sort((a, b) => b[1] - a[1])
            .map(([k, count]) => {
              const hidden = hiddenNodeTypes.has(k);
              return (
                <button
                  key={k}
                  type="button"
                  onClick={() =>
                    setHiddenNodeTypes(togglePill(hiddenNodeTypes, k))
                  }
                  className={`rounded-full border px-2 py-0.5 text-xs ${
                    hidden
                      ? "border-gray-300 text-gray-400 line-through"
                      : "border-gray-700 text-gray-700"
                  }`}
                >
                  {k} ({count})
                </button>
              );
            })}
        </div>
      </div>
      <div className="rounded-md border bg-white p-3">
        <h3 className="mb-2 text-xs font-semibold uppercase text-gray-500">
          Relationships
        </h3>
        <div className="flex flex-wrap gap-1.5">
          {[...edgeTypeCounts.entries()]
            .sort((a, b) => b[1] - a[1])
            .map(([k, count]) => {
              const hidden = hiddenEdgeTypes.has(k);
              return (
                <button
                  key={k}
                  type="button"
                  onClick={() =>
                    setHiddenEdgeTypes(togglePill(hiddenEdgeTypes, k))
                  }
                  className={`rounded-full border px-2 py-0.5 text-xs ${
                    hidden
                      ? "border-gray-300 text-gray-400 line-through"
                      : "border-gray-700 text-gray-700"
                  }`}
                >
                  {k} ({count})
                </button>
              );
            })}
        </div>
      </div>
    </div>
  );
}

interface DetailPanelProps {
  node: {
    uuid: string;
    label: string;
    type?: string | null;
    name?: string | null;
    summary?: string | null;
  } | null;
  neighbors: {
    edge: {
      label: string;
      source: string;
      target: string;
      name?: string | null;
      fact?: string | null;
    };
    otherUuid: string;
  }[];
  onClear: () => void;
}

function DetailPanel({ node, neighbors, onClear }: DetailPanelProps) {
  return (
    <div className="col-span-12 md:col-span-2">
      <div className="rounded-md border bg-white p-3 text-sm">
        {!node ? (
          <div className="text-gray-500">Click a node to inspect.</div>
        ) : (
          <>
            <div className="flex items-start justify-between gap-2">
              <div className="min-w-0">
                <div className="truncate text-base font-semibold">
                  {node.name ?? node.uuid.slice(0, 12)}
                </div>
                <div className="text-xs text-gray-500">
                  {node.type ?? node.label}
                </div>
              </div>
              <button
                type="button"
                onClick={onClear}
                className="text-xs text-gray-400 hover:text-gray-600"
                aria-label="Clear selection"
              >
                ✕
              </button>
            </div>
            <div className="mt-2 font-mono text-[10px] text-gray-400">
              {node.uuid}
            </div>
            {node.summary && (
              <div className="mt-3 whitespace-pre-wrap text-xs text-gray-700">
                {node.summary}
              </div>
            )}
            <h4 className="mt-4 text-xs font-semibold uppercase text-gray-500">
              {neighbors.length} edges
            </h4>
            <ul className="mt-1 max-h-72 space-y-1 overflow-y-auto text-xs">
              {neighbors.map(({ edge, otherUuid }) => (
                <li key={edge.label + otherUuid} className="border-l-2 border-gray-200 pl-2">
                  <div className="text-gray-700">
                    <span className="font-medium">{edge.label}</span>
                    {edge.name && (
                      <span className="text-gray-500"> · {edge.name}</span>
                    )}
                  </div>
                  {edge.fact && (
                    <div className="text-gray-600">{edge.fact}</div>
                  )}
                  <div className="font-mono text-[10px] text-gray-400">
                    → {otherUuid.slice(0, 12)}
                  </div>
                </li>
              ))}
            </ul>
          </>
        )}
      </div>
    </div>
  );
}
