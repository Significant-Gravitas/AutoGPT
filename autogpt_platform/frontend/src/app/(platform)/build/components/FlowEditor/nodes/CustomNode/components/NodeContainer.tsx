import { cn } from "@/lib/utils";
import { nodeStyleBasedOnStatus } from "../helpers";

import { useNodeStore } from "@/app/(platform)/build/stores/nodeStore";

export const NodeContainer = ({
  children,
  nodeId,
  selected,
}: {
  children: React.ReactNode;
  nodeId: string;
  selected: boolean;
}) => {
  const status = useNodeStore((state) => state.getNodeStatus(nodeId));
  return (
    <div
      className={cn(
        "z-12 max-w-[370px] rounded-xlarge shadow-lg shadow-slate-900/5 ring-1 ring-slate-200/60 backdrop-blur-sm",
        selected && "shadow-2xl ring-2 ring-slate-200",
        status && nodeStyleBasedOnStatus[status],
      )}
    >
      {children}
    </div>
  );
};
