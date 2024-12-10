import React from "react";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { ExecutionMeta } from "@/lib/autogpt-server-api";

export const FlowRunStatusBadge: React.FC<{
  status: ExecutionMeta["status"];
  className?: string;
}> = ({ status, className }) => (
  <Badge
    variant="default"
    className={cn(
      status === "running"
        ? "bg-blue-500 dark:bg-blue-700"
        : status === "waiting"
          ? "bg-yellow-500 dark:bg-yellow-600"
          : status === "success"
            ? "bg-green-500 dark:bg-green-600"
            : "bg-red-500 dark:bg-red-700",
      className,
    )}
  >
    {status}
  </Badge>
);
