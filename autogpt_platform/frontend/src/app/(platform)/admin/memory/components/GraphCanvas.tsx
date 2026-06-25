"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import dynamic from "next/dynamic";
import type { GraphNode } from "@/app/api/__generated__/models/graphNode";
import type { GraphEdge } from "@/app/api/__generated__/models/graphEdge";

// react-force-graph-2d uses HTMLCanvas + window APIs at import time, so it
// can't run during Next's server rendering pass.
const ForceGraph2D = dynamic(() => import("react-force-graph-2d"), {
  ssr: false,
  loading: () => (
    <div className="flex h-full items-center justify-center text-sm text-gray-500">
      Loading graph engine…
    </div>
  ),
});

// Color palette per node "type" (custom-typed entities) / "label" fallback.
// Mirrors the FalkorDB-browser palette in spirit — distinct, saturated,
// readable on white.
const TYPE_COLORS: Record<string, string> = {
  Person: "#7dd3fc", // sky-300
  Organization: "#86efac", // green-300
  Project: "#fde68a", // amber-200
  Concept: "#fca5a5", // red-300
  Preference: "#c4b5fd", // violet-300
  Rule: "#bef264", // lime-300
  Episodic: "#fdba74", // orange-300
  Community: "#93c5fd", // blue-300
  Entity: "#f0abfc", // fuchsia-300 — default for un-typed entities
};

const EDGE_COLORS: Record<string, string> = {
  RELATES_TO: "#a855f7", // purple-500
  MENTIONS: "#6366f1", // indigo-500
  HAS_MEMBER: "#ec4899", // pink-500
};

export interface GraphCanvasProps {
  nodes: GraphNode[];
  edges: GraphEdge[];
  hiddenNodeTypes: Set<string>;
  hiddenEdgeTypes: Set<string>;
  selectedUuid: string | null;
  onSelect: (uuid: string | null) => void;
}

type CanvasNode = GraphNode & {
  // Position fields written back by d3-force; declared here so TS is happy.
  x?: number;
  y?: number;
  fx?: number;
  fy?: number;
};

type CanvasLink = {
  source: string | CanvasNode;
  target: string | CanvasNode;
  uuid: string;
  label: string;
  name?: string | null;
};

// force-graph hands tooltip labels to float-tooltip, which injects them
// via innerHTML. Node/edge names are LLM-extracted from conversation
// content (which can carry attacker-controlled material), so everything
// that reaches a label accessor must be HTML-escaped first.
export function escapeHtml(value: string): string {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

export function getNodeTooltipLabel(n: GraphNode): string {
  return escapeHtml(`${n.type ?? n.label}: ${n.name ?? n.uuid.slice(0, 8)}`);
}

export function getLinkTooltipLabel(l: CanvasLink): string {
  return escapeHtml(l.name ? `${l.label}: ${l.name}` : l.label);
}

// d3-force replaces link ``source``/``target`` string ids with node
// object references once the simulation initializes, so both shapes
// must be handled when matching a link against the selected node.
export function linkTouchesNode(
  l: CanvasLink,
  selectedUuid: string | null,
): boolean {
  if (!selectedUuid) return false;
  const sourceId = typeof l.source === "object" ? l.source.uuid : l.source;
  const targetId = typeof l.target === "object" ? l.target.uuid : l.target;
  return sourceId === selectedUuid || targetId === selectedUuid;
}

export function GraphCanvas({
  nodes,
  edges,
  hiddenNodeTypes,
  hiddenEdgeTypes,
  selectedUuid,
  onSelect,
}: GraphCanvasProps) {
  const fgRef = useRef<unknown>(null);
  const wrapperRef = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState({ width: 800, height: 600 });

  // Track wrapper size so the canvas fills its container.
  useEffect(() => {
    if (!wrapperRef.current) return;
    const el = wrapperRef.current;
    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        setSize({ width: Math.floor(width), height: Math.floor(height) });
      }
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const data = useMemo(() => {
    function classify(n: GraphNode): string {
      return n.type ?? n.label;
    }
    const visibleNodes = nodes.filter((n) => !hiddenNodeTypes.has(classify(n)));
    const visibleNodeIds = new Set(visibleNodes.map((n) => n.uuid));
    const visibleEdges = edges
      .filter((e) => !hiddenEdgeTypes.has(e.label))
      .filter(
        (e) => visibleNodeIds.has(e.source) && visibleNodeIds.has(e.target),
      )
      .map((e) => ({
        source: e.source,
        target: e.target,
        uuid: e.uuid,
        label: e.label,
        name: e.name,
      }));
    return {
      nodes: visibleNodes as CanvasNode[],
      links: visibleEdges as CanvasLink[],
    };
  }, [nodes, edges, hiddenNodeTypes, hiddenEdgeTypes]);

  return (
    <div ref={wrapperRef} className="h-[70vh] w-full">
      {/* react-force-graph-2d's generic types collapse to a loose
          `{ [k: string]: any }` shape when reached through Next's dynamic
          import, so the accessor callbacks are cast to `never` and narrowed
          inside. Same pattern as `graphData={data as never}`. */}
      <ForceGraph2D
        ref={fgRef as never}
        width={size.width}
        height={size.height}
        graphData={data as never}
        nodeId="uuid"
        nodeLabel={getNodeTooltipLabel as never}
        nodeRelSize={4}
        nodeVal={
          ((n: CanvasNode) => (n.uuid === selectedUuid ? 6 : 3)) as never
        }
        nodeColor={
          ((n: CanvasNode) =>
            TYPE_COLORS[n.type ?? n.label] ?? TYPE_COLORS.Entity) as never
        }
        nodeCanvasObject={
          ((n: CanvasNode, ctx: CanvasRenderingContext2D, scale: number) => {
            const r = (n.uuid === selectedUuid ? 6 : 3) * 0.8;
            const color = TYPE_COLORS[n.type ?? n.label] ?? TYPE_COLORS.Entity;
            ctx.beginPath();
            ctx.arc(n.x ?? 0, n.y ?? 0, r, 0, 2 * Math.PI);
            ctx.fillStyle = color;
            ctx.fill();
            if (n.uuid === selectedUuid) {
              ctx.lineWidth = 1.5 / scale;
              ctx.strokeStyle = "#111827";
              ctx.stroke();
            }
            if (scale > 1.6 && n.name) {
              const label =
                n.name.length > 28 ? `${n.name.slice(0, 27)}…` : n.name;
              ctx.font = `${10 / scale}px sans-serif`;
              ctx.fillStyle = "#111827";
              ctx.textAlign = "center";
              ctx.textBaseline = "top";
              ctx.fillText(label, n.x ?? 0, (n.y ?? 0) + r + 1.5 / scale);
            }
          }) as never
        }
        linkColor={
          ((l: CanvasLink) => EDGE_COLORS[l.label] ?? "#94a3b8") as never
        }
        linkLabel={getLinkTooltipLabel as never}
        linkWidth={
          ((l: CanvasLink) =>
            linkTouchesNode(l, selectedUuid) ? 2 : 0.6) as never
        }
        linkDirectionalArrowLength={3}
        linkDirectionalArrowRelPos={0.95}
        onNodeClick={((n: CanvasNode) => onSelect(n.uuid)) as never}
        onBackgroundClick={() => onSelect(null)}
        cooldownTicks={120}
        d3VelocityDecay={0.3}
      />
    </div>
  );
}
