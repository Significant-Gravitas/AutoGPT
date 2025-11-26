import { cn } from "@/lib/utils";
import { nodeStyleBasedOnStatus } from "../helpers";

import { useNodeStore } from "@/app/(platform)/build/stores/nodeStore";
import { useShallow } from "zustand/react/shallow";

export const NodeContainer = ({
  children,
  nodeId,
  selected,
}: {
  children: React.ReactNode;
  nodeId: string;
  selected: boolean;
}) => {
  const status = useNodeStore(
    useShallow((state) => state.getNodeStatus(nodeId)),
  );
  return (
    <div
      className={cn(
        "z-12 max-w-[370px] rounded-xlarge ring-1 ring-slate-200/60",
        selected && "shadow-lg ring-2 ring-slate-200",
        status && nodeStyleBasedOnStatus[status],
      )}
    >
      {children}
    </div>
  );
};
