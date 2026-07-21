import React from "react";

import { Badge } from "@/components/__legacy__/ui/badge";

export type AgentStatus = "active" | "inactive" | "error" | "broken";

const statusData: Record<
  AgentStatus,
  { label: string; variant: keyof typeof statusStyles }
> = {
  active: { label: "Active", variant: "success" },
  error: { label: "Error", variant: "destructive" },
  broken: { label: "Broken", variant: "destructive" },
  inactive: { label: "Inactive", variant: "secondary" },
};

const statusStyles = {
  success:
    "bg-green-100 text-green-800 hover:bg-green-100 hover:text-green-800",
  destructive: "bg-red-100 text-red-800 hover:bg-red-100 hover:text-red-800",
  warning:
    "bg-yellow-100 text-yellow-800 hover:bg-yellow-100 hover:text-yellow-800",
  info: "bg-blue-100 text-blue-800 hover:bg-blue-100 hover:text-blue-800",
  secondary:
    "bg-slate-100 text-slate-800 hover:bg-slate-100 hover:text-slate-800",
};

export function AgentStatusChip({
  status,
}: {
  status: AgentStatus;
}): React.ReactElement {
  return (
    <Badge
      variant="secondary"
      className={`text-xs font-medium ${statusStyles[statusData[status]?.variant]} rounded-[45px] px-[9px] py-[3px]`}
    >
      {statusData[status]?.label}
    </Badge>
  );
}
