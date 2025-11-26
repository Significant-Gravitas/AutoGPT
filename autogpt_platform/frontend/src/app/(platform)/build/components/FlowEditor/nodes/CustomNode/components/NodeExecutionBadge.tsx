import { useNodeStore } from "@/app/(platform)/build/stores/nodeStore";
import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import { Badge } from "@/components/__legacy__/ui/badge";
import { LoadingSpinner } from "@/components/__legacy__/ui/loading";
import { cn } from "@/lib/utils";
import { useShallow } from "zustand/react/shallow";

const statusStyles: Record<AgentExecutionStatus, string> = {
  INCOMPLETE: "text-slate-700 border-slate-400",
  QUEUED: "text-blue-700 border-blue-400",
  RUNNING: "text-amber-700 border-amber-400",
  COMPLETED: "text-green-700 border-green-400",
  TERMINATED: "text-orange-700 border-orange-400",
  FAILED: "text-red-700  border-red-400",
};

export const NodeExecutionBadge = ({ nodeId }: { nodeId: string }) => {
  const status = useNodeStore(
    useShallow((state) => state.getNodeStatus(nodeId)),
  );
  if (!status) return null;
  return (
    <div className="flex items-center justify-end rounded-b-xl py-2 pr-4">
      <Badge
        className={cn(statusStyles[status], "gap-2 rounded-full bg-white")}
      >
        {status}
        {status === AgentExecutionStatus.RUNNING && (
          <LoadingSpinner className="size-4" />
        )}
      </Badge>
    </div>
  );
};
