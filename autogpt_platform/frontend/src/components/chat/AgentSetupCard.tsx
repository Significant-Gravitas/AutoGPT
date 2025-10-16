"use client";

import React from "react";
import { Button } from "@/components/atoms/Button/Button";
import {
  CheckCircle,
  Calendar,
  Webhook,
  ExternalLink,
  Clock,
  PlayCircle,
  Library,
} from "lucide-react";
import { cn } from "@/lib/utils";

interface AgentSetupCardProps {
  status: string;
  triggerType: "schedule" | "webhook";
  name: string;
  graphId: string;
  graphVersion: number;
  scheduleId?: string;
  webhookUrl?: string;
  cron?: string;
  _cronUtc?: string;
  timezone?: string;
  nextRun?: string;
  addedToLibrary?: boolean;
  libraryId?: string;
  message: string;
  className?: string;
}

export function AgentSetupCard({
  status,
  triggerType,
  name,
  graphId,
  graphVersion,
  scheduleId,
  webhookUrl,
  cron,
  _cronUtc,
  timezone,
  nextRun,
  addedToLibrary,
  libraryId,
  message,
  className,
}: AgentSetupCardProps) {
  const isSuccess = status === "success";

  const formatNextRun = (isoString: string) => {
    try {
      const date = new Date(isoString);
      return date.toLocaleString();
    } catch {
      return isoString;
    }
  };

  const handleViewInLibrary = () => {
    if (libraryId) {
      window.open(`/library/agents/${libraryId}`, "_blank");
    } else {
      window.open(`/library`, "_blank");
    }
  };

  const handleViewRuns = () => {
    if (scheduleId) {
      window.open(`/library/runs?scheduleId=${scheduleId}`, "_blank");
    } else {
      window.open(`/library/runs`, "_blank");
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text).then(() => {
      // Could add a toast notification here
      console.log("Copied to clipboard:", text);
    });
  };

  return (
    <div
      className={cn(
        "my-4 overflow-hidden rounded-lg border",
        isSuccess
          ? "border-green-200 bg-gradient-to-br from-green-50 to-emerald-50 dark:border-green-800 dark:from-green-950/30 dark:to-emerald-950/30"
          : "border-red-200 bg-gradient-to-br from-red-50 to-rose-50 dark:border-red-800 dark:from-red-950/30 dark:to-rose-950/30",
        "duration-500 animate-in fade-in-50 slide-in-from-bottom-2",
        className,
      )}
    >
      <div className="px-6 py-5">
        <div className="mb-4 flex items-center gap-3">
          <div
            className={cn(
              "flex h-10 w-10 items-center justify-center rounded-full",
              isSuccess ? "bg-green-600" : "bg-red-600",
            )}
          >
            {isSuccess ? (
              <CheckCircle className="h-5 w-5 text-white" />
            ) : (
              <ExternalLink className="h-5 w-5 text-white" />
            )}
          </div>
          <div>
            <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
              {isSuccess ? "Agent Setup Complete" : "Setup Failed"}
            </h3>
            <p className="text-sm text-neutral-600 dark:text-neutral-400">
              {name}
            </p>
          </div>
        </div>

        {/* Success message */}
        <div className="mb-5 rounded-md bg-white/50 p-4 dark:bg-neutral-900/50">
          <p className="text-sm text-neutral-700 dark:text-neutral-300">
            {message}
          </p>
        </div>

        {/* Setup details */}
        {isSuccess && (
          <div className="mb-5 space-y-3">
            {/* Trigger type badge */}
            <div className="flex items-center gap-2">
              {triggerType === "schedule" ? (
                <>
                  <Calendar className="h-4 w-4 text-blue-600" />
                  <span className="text-sm font-medium text-blue-700 dark:text-blue-400">
                    Scheduled Execution
                  </span>
                </>
              ) : (
                <>
                  <Webhook className="h-4 w-4 text-purple-600" />
                  <span className="text-sm font-medium text-purple-700 dark:text-purple-400">
                    Webhook Trigger
                  </span>
                </>
              )}
            </div>

            {/* Schedule details */}
            {triggerType === "schedule" && cron && (
              <div className="space-y-2 rounded-md bg-blue-50 p-3 text-sm dark:bg-blue-950/30">
                <div className="flex items-center justify-between">
                  <span className="text-neutral-600 dark:text-neutral-400">
                    Schedule:
                  </span>
                  <code className="rounded bg-neutral-200 px-2 py-0.5 font-mono text-xs dark:bg-neutral-800">
                    {cron}
                  </code>
                </div>
                {timezone && (
                  <div className="flex items-center justify-between">
                    <span className="text-neutral-600 dark:text-neutral-400">
                      Timezone:
                    </span>
                    <span className="font-medium">{timezone}</span>
                  </div>
                )}
                {nextRun && (
                  <div className="flex items-center gap-2">
                    <Clock className="h-4 w-4 text-blue-600" />
                    <span className="text-neutral-600 dark:text-neutral-400">
                      Next run:
                    </span>
                    <span className="font-medium">
                      {formatNextRun(nextRun)}
                    </span>
                  </div>
                )}
              </div>
            )}

            {/* Webhook details */}
            {triggerType === "webhook" && webhookUrl && (
              <div className="space-y-2 rounded-md bg-purple-50 p-3 dark:bg-purple-950/30">
                <div className="flex items-center justify-between">
                  <span className="text-sm text-neutral-600 dark:text-neutral-400">
                    Webhook URL:
                  </span>
                  <button
                    onClick={() => copyToClipboard(webhookUrl)}
                    className="text-xs text-purple-600 hover:text-purple-700 hover:underline dark:text-purple-400"
                  >
                    Copy
                  </button>
                </div>
                <code className="block break-all rounded bg-neutral-200 p-2 font-mono text-xs dark:bg-neutral-800">
                  {webhookUrl}
                </code>
              </div>
            )}

            {/* Library status */}
            {addedToLibrary && (
              <div className="flex items-center gap-2 text-sm text-green-700 dark:text-green-400">
                <Library className="h-4 w-4" />
                <span>Added to your library</span>
              </div>
            )}
          </div>
        )}

        {/* Action buttons */}
        {isSuccess && (
          <div className="flex gap-3">
            <Button
              onClick={handleViewInLibrary}
              variant="primary"
              size="small"
              className="flex-1"
            >
              <Library className="mr-2 h-4 w-4" />
              View in Library
            </Button>
            {triggerType === "schedule" && (
              <Button
                onClick={handleViewRuns}
                variant="secondary"
                size="small"
                className="flex-1"
              >
                <PlayCircle className="mr-2 h-4 w-4" />
                View Runs
              </Button>
            )}
          </div>
        )}

        {/* Additional info */}
        <div className="mt-4 space-y-1 text-xs text-neutral-500 dark:text-neutral-500">
          <p>
            Agent ID: <span className="font-mono">{graphId}</span>
          </p>
          <p>Version: {graphVersion}</p>
          {scheduleId && (
            <p>
              Schedule ID: <span className="font-mono">{scheduleId}</span>
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
