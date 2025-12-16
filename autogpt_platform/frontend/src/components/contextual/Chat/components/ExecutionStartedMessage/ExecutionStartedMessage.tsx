import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { ArrowSquareOut, CheckCircle, Play } from "@phosphor-icons/react";

export interface ExecutionStartedMessageProps {
  executionId: string;
  agentName?: string;
  message?: string;
  onViewExecution?: () => void;
  className?: string;
}

export function ExecutionStartedMessage({
  executionId,
  agentName,
  message = "Agent execution started successfully",
  onViewExecution,
  className,
}: ExecutionStartedMessageProps) {
  return (
    <div
      className={cn(
        "mx-4 my-2 flex flex-col gap-4 rounded-lg border border-green-200 bg-green-50 p-6",
        className,
      )}
    >
      {/* Icon & Header */}
      <div className="flex items-start gap-4">
        <div className="flex h-12 w-12 flex-shrink-0 items-center justify-center rounded-full bg-green-500">
          <CheckCircle size={24} weight="bold" className="text-white" />
        </div>
        <div className="flex-1">
          <Text variant="h3" className="mb-1 text-green-900">
            Execution Started
          </Text>
          <Text variant="body" className="text-green-700">
            {message}
          </Text>
        </div>
      </div>

      {/* Details */}
      <div className="rounded-md bg-green-100 p-4">
        <div className="space-y-2">
          {agentName && (
            <div className="flex items-center justify-between">
              <Text variant="small" className="font-semibold text-green-900">
                Agent:
              </Text>
              <Text variant="body" className="text-green-800">
                {agentName}
              </Text>
            </div>
          )}
          <div className="flex items-center justify-between">
            <Text variant="small" className="font-semibold text-green-900">
              Execution ID:
            </Text>
            <Text variant="small" className="font-mono text-green-800">
              {executionId.slice(0, 16)}...
            </Text>
          </div>
        </div>
      </div>

      {/* Action Buttons */}
      {onViewExecution && (
        <div className="flex gap-3">
          <Button
            onClick={onViewExecution}
            variant="primary"
            className="flex flex-1 items-center justify-center gap-2"
          >
            <ArrowSquareOut size={20} weight="bold" />
            View Execution
          </Button>
        </div>
      )}

      <div className="flex items-center gap-2 text-green-600">
        <Play size={16} weight="fill" />
        <Text variant="small">
          Your agent is now running. You can monitor its progress in the monitor
          page.
        </Text>
      </div>
    </div>
  );
}
