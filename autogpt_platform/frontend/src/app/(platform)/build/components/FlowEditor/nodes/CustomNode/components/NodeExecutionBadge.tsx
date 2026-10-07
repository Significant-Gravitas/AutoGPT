import { useNodeStore } from "@/app/(platform)/build/stores/nodeStore";
import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import { Badge } from "@/components/atoms/Badge/Badge";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { cn } from "@/lib/utils";
import { useShallow } from "zustand/react/shallow";

const statusStyles: Record<AgentExecutionStatus, string> = {
  INCOMPLETE: "text-slate-700 border-slate-400",
  QUEUED: "text-blue-700 border-blue-400",
  RUNNING: "text-yellow-700 border-yellow-400",
  REVIEW: "text-yellow-700 border-yellow-400 bg-yellow-50",
  COMPLETED: "text-green-700 border-green-400",
  TERMINATED: "text-orange-700 border-orange-400",
  FAILED: "text-red-700 border-red-400",
};

export const NodeExecutionBadge = ({ nodeId }: { nodeId: string }) => {
  const status = useNodeStore(
    useShallow((state) => state.getNodeStatus(nodeId)),
  );
  if (!status) return null;
  return (
    <div className="flex items-center justify-end rounded-b-xl py-2 pr-4">
      <Badge
        variant="info"
        className={cn(
          "gap-2 rounded-full border bg-white px-2.5 font-semibold ring-0",
          statusStyles[status],
        )}
      >
        {status}
        {status === AgentExecutionStatus.RUNNING && (
          <LoadingSpinner size="small" />
        )}
      </Badge>
    </div>
  );
};
