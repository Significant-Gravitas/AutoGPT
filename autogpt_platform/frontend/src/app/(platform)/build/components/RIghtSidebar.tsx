import { useMemo } from "react";

import { Link } from "@/app/api/__generated__/models/link";
import { useEdgeStore } from "../stores/edgeStore";
import { useNodeStore } from "../stores/nodeStore";
import { scrollbarStyles } from "@/components/styles/scrollbars";
import { cn } from "@/lib/utils";

export const RightSidebar = () => {
  const connections = useEdgeStore((s) => s.connections);
  const nodes = useNodeStore((s) => s.nodes);

  const backendLinks: Link[] = useMemo(
    () =>
      connections.map((c) => ({
        source_id: c.source,
        sink_id: c.target,
        source_name: c.sourceHandle,
        sink_name: c.targetHandle,
      })),
    [connections],
  );

  return (
    <div
      className={cn(
        "flex h-full w-full flex-col border-l border-slate-200 bg-white p-4 dark:border-slate-700 dark:bg-slate-900",
        scrollbarStyles,
      )}
    >
      <div className="mb-4">
        <h2 className="text-lg font-semibold text-slate-800 dark:text-slate-200">
          Flow Debug Panel
        </h2>
      </div>

      <div className="flex-1 overflow-y-auto">
        <h3 className="mb-2 text-sm font-semibold text-slate-700 dark:text-slate-200">
          Nodes ({nodes.length})
        </h3>
        <div className="mb-6 space-y-3">
          {nodes.map((n) => (
            <div
              key={n.id}
              className="rounded border p-2 text-xs dark:border-slate-700"
            >
              <div className="mb-1 font-medium">
                #{n.id} {n.data?.title ? `– ${n.data.title}` : ""}
              </div>
              <div className="text-slate-500 dark:text-slate-400">
                hardcodedValues
              </div>
              <pre className="mt-1 max-h-40 overflow-auto rounded bg-slate-50 p-2 dark:bg-slate-800">
                {JSON.stringify(n.data?.hardcodedValues ?? {}, null, 2)}
              </pre>
            </div>
          ))}
        </div>

        <h3 className="mb-2 text-sm font-semibold text-slate-700 dark:text-slate-200">
          Links ({backendLinks.length})
        </h3>
        <div className="mb-6 space-y-3">
          {connections.map((c) => (
            <div
              key={c.edge_id}
              className="rounded border p-2 text-xs dark:border-slate-700"
            >
              <div className="font-medium">
                {c.source}[{c.sourceHandle}] → {c.target}[{c.targetHandle}]
              </div>
              <div className="mt-1 text-slate-500 dark:text-slate-400">
                edge_id: {c.edge_id}
              </div>
            </div>
          ))}
        </div>

        <h4 className="mb-2 text-xs font-semibold text-slate-600 dark:text-slate-300">
          Backend Links JSON
        </h4>
        <pre className="max-h-64 overflow-auto rounded bg-slate-50 p-2 text-[11px] dark:bg-slate-800">
          {JSON.stringify(backendLinks, null, 2)}
        </pre>
      </div>
    </div>
  );
};
