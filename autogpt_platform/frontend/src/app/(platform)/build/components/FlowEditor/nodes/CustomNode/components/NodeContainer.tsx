import { cn } from "@/lib/utils";
import { nodeStyleBasedOnStatus } from "../helpers";

import { useNodeStore } from "@/app/(platform)/build/stores/nodeStore";
import { useShallow } from "zustand/react/shallow";
import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";

export const NodeContainer = ({
  children,
  nodeId,
  selected,
  hasErrors, // these are configuration errors that occur before executing the graph -- more like validation errors
}: {
  children: React.ReactNode;
  nodeId: string;
  selected: boolean;
  hasErrors?: boolean;
}) => {
  const status = useNodeStore(
    useShallow((state) => state.getNodeStatus(nodeId)),
  );
  return (
    <div
      className={cn(
        "z-12 w-[350px] rounded-xlarge ring-1 ring-slate-200/60",
        selected && "shadow-lg ring-2 ring-slate-200",
        status && nodeStyleBasedOnStatus[status],
        hasErrors ? nodeStyleBasedOnStatus[AgentExecutionStatus.FAILED] : "",
      )}
    >
      {children}
    </div>
  );
};
