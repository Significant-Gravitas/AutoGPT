// autogpt_platform/frontend/src/app/(platform)/build/components/FlowEditor/RightSidebar.tsx
import { useMemo, useState } from "react";

import { Link } from "@/app/api/__generated__/models/link";
import { useEdgeStore } from "./store/edgeStore";
import { useNodeStore } from "./store/nodeStore";
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import { Button } from "@/components/atoms/Button/Button";
import { scrollbarStyles } from "@/components/styles/scrollbars";
import { cn } from "@/lib/utils";

export const RightSidebar = () => {
  const [isOpen, setIsOpen] = useState(false);
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
    <Sheet open={isOpen} onOpenChange={setIsOpen}>
      <SheetTrigger asChild>
        <Button variant="secondary" className="absolute right-4 top-4 z-50">
          Debug Panel
        </Button>
      </SheetTrigger>
      <SheetContent
        side="right"
        className={cn("w-96 overflow-y-auto", scrollbarStyles)}
      >
        <SheetHeader>
          <SheetTitle>Flow Debug Panel</SheetTitle>
        </SheetHeader>

        <div className="mt-6">
          <h3 className="mb-2 text-sm font-semibold text-slate-700 dark:text-slate-200">
            Nodes ({nodes.length})
          </h3>
          <div className="space-y-3">
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

          <h3 className="mb-2 mt-6 text-sm font-semibold text-slate-700 dark:text-slate-200">
            Links ({backendLinks.length})
          </h3>
          <div className="space-y-3">
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

          <h4 className="mb-2 mt-6 text-xs font-semibold text-slate-600 dark:text-slate-300">
            Backend Links JSON
          </h4>
          <pre className="max-h-64 overflow-auto rounded bg-slate-50 p-2 text-[11px] dark:bg-slate-800">
            {JSON.stringify(backendLinks, null, 2)}
          </pre>
        </div>
      </SheetContent>
    </Sheet>
  );
};
