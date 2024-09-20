import React from "react";
import { FlowRun } from "@/lib/types";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";

export const FlowRunStatusBadge: React.FC<{
  status: FlowRun["status"];
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
