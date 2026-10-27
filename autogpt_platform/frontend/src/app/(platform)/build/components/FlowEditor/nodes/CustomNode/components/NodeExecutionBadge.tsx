import { useNodeStore } from "@/app/(platform)/build/stores/nodeStore";
import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import { Badge } from "@/components/atoms/Badge/Badge";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { cn } from "@/lib/utils";
import { useShallow } from "zustand/react/shallow";

const statusStyles: Record<AgentExecutionStatus, string> = {
  // TODO(tokens): no exact role: text-slate-700 #475263, border-slate-400 #8A97A8
  INCOMPLETE: "text-slate-700 border-slate-400",
  // TODO(tokens): no exact role: text-blue-700 #1D4ED8, border-blue-400 #60A5FA
  QUEUED: "text-blue-700 border-blue-400",
  // TODO(tokens): no fitting role for text-amber-700 #B45309; #B45309 is warning-foreground (a status colour), and a running node is in progress, not a caution; no role marks in progress (QUALITY 41(c)); no exact role: border-amber-400 #FBBF24
  RUNNING: "text-amber-700 border-amber-400",
  // TODO(tokens): no exact role: text-yellow-700 #AE8800, border-yellow-400 #F7CD33, bg-yellow-50 #FEF9E6
  REVIEW: "text-yellow-700 border-yellow-400 bg-yellow-50",
  // TODO(tokens): no exact role: border-green-400 #45B56E
  COMPLETED: "text-success-foreground border-green-400",
  // TODO(tokens): no exact role: text-orange-700 #B46000, border-orange-400 #FE9F33
  TERMINATED: "text-orange-700 border-orange-400",
  // TODO(tokens): no exact role: border-red-400 #F26969
  FAILED: "text-error-foreground  border-red-400",
};

export const NodeExecutionBadge = ({ nodeId }: { nodeId: string }) => {
  const status = useNodeStore(
    useShallow((state) => state.getNodeStatus(nodeId)),
  );
  if (!status) return null;
  return (
    <div className="flex items-center justify-end rounded-b-xl py-2 pr-4">
      <Badge
        variant="solid"
        className={cn(statusStyles[status], "gap-2 rounded-full bg-surface")}
      >
        {status}
        {status === AgentExecutionStatus.RUNNING && (
          <LoadingSpinner
            variant="circle"
            className="size-4"
            aria-hidden="true"
          />
        )}
      </Badge>
    </div>
  );
};
