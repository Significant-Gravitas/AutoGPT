import { useNodeStore } from "@/app/(platform)/build/stores/nodeStore";
import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import { Badge } from "@/components/atoms/Badge/Badge";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { cn } from "@/lib/utils";
import { useShallow } from "zustand/react/shallow";

const statusStyles: Record<AgentExecutionStatus, string> = {
  INCOMPLETE: "text-foreground-body border-border-strong",
  QUEUED: "text-info-foreground border-info",
  RUNNING: "text-warning-foreground border-warning",
  // oxlint-disable-next-line design-system/no-raw-colors -- mustard yellow has no role; the warning roles are amber, deferred until approved
  REVIEW: "text-yellow-700 border-yellow-400 bg-yellow-50",
  COMPLETED: "text-success-foreground border-success",
  // oxlint-disable-next-line design-system/no-raw-colors -- orange has no role; mapping it to warning (amber) is deferred until approved
  TERMINATED: "text-orange-700 border-orange-400",
  FAILED: "text-error-foreground  border-error",
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
        // oxlint-disable-next-line design-system/require-static-classes -- statusStyles maps each status to whole, static class strings
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
