import { AgentExecutionStatus } from "@/app/api/__generated__/models/agentExecutionStatus";
import { Badge } from "@/components/__legacy__/ui/badge";
import { cn } from "@/lib/utils";

const statusStyles: Record<AgentExecutionStatus, string> = {
  INCOMPLETE: "text-slate-700 border-slate-400",
  QUEUED: "text-blue-700 border-blue-400",
  RUNNING: "text-amber-700 border-amber-400",
  COMPLETED: "text-green-700 border-green-400",
  TERMINATED: "text-orange-700 border-orange-400",
  FAILED: "text-red-700  border-red-400",
};

export const NodeExecutionBadge = ({
  status,
}: {
  status: AgentExecutionStatus;
}) => {
  return (
    <div className="flex items-center justify-end rounded-b-xl py-2 pr-4">
      <Badge className={cn(statusStyles[status], "rounded-full bg-white")}>
        {status}
      </Badge>
    </div>
  );
};
