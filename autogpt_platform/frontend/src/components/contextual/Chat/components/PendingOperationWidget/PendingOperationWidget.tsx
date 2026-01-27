"use client";

import { Card } from "@/components/atoms/Card/Card";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { CircleNotch, CheckCircle, XCircle } from "@phosphor-icons/react";

type OperationStatus =
  | "pending"
  | "started"
  | "in_progress"
  | "completed"
  | "error";

interface Props {
  status: OperationStatus;
  message: string;
  toolName?: string;
  className?: string;
}

export function PendingOperationWidget({
  status,
  message,
  toolName,
  className,
}: Props) {
  const isPending =
    status === "pending" || status === "started" || status === "in_progress";
  const isCompleted = status === "completed";
  const isError = status === "error";

  return (
    <div
      className={cn(
        "group relative flex w-full justify-start gap-3 px-4 py-3",
        className,
      )}
    >
      <div className="flex w-full max-w-3xl gap-3">
        <div className="flex-shrink-0">
          <div
            className={cn(
              "flex h-7 w-7 items-center justify-center rounded-lg",
              isPending && "bg-blue-500",
              isCompleted && "bg-green-500",
              isError && "bg-red-500",
            )}
          >
            {isPending && (
              <CircleNotch
                className="h-4 w-4 animate-spin text-white"
                weight="bold"
              />
            )}
            {isCompleted && (
              <CheckCircle className="h-4 w-4 text-white" weight="bold" />
            )}
            {isError && (
              <XCircle className="h-4 w-4 text-white" weight="bold" />
            )}
          </div>
        </div>

        <div className="flex min-w-0 flex-1 flex-col">
          <Card className="space-y-2 p-4">
            <div>
              <Text variant="h4" className="mb-1 text-slate-900">
                {isPending && "Creating Agent"}
                {isCompleted && "Operation Complete"}
                {isError && "Operation Failed"}
              </Text>
              <Text variant="small" className="text-slate-600">
                {message}
              </Text>
            </div>

            {isPending && (
              <Text variant="small" className="italic text-slate-500">
                Check your library in a few minutes.
              </Text>
            )}

            {toolName && (
              <Text variant="small" className="text-slate-400">
                Tool: {toolName}
              </Text>
            )}
          </Card>
        </div>
      </div>
    </div>
  );
}
