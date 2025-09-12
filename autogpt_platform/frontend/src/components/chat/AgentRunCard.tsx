"use client";

import React from "react";
import { Button } from "@/components/atoms/Button/Button";
import { Play, ExternalLink, Clock, CheckCircle, XCircle, Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";
import Link from "next/link";

interface AgentRunCardProps {
  executionId: string;
  graphId: string;
  graphName: string;
  status: string;
  inputs?: Record<string, any>;
  outputs?: Record<string, any>;
  error?: string;
  timeoutReached?: boolean;
  endedAt?: string;
  className?: string;
}

export function AgentRunCard({
  executionId,
  graphId,
  graphName,
  status,
  inputs,
  outputs,
  error,
  timeoutReached,
  endedAt,
  className,
}: AgentRunCardProps) {
  const getStatusIcon = () => {
    switch (status) {
      case "COMPLETED":
        return <CheckCircle className="h-5 w-5 text-green-600" />;
      case "FAILED":
      case "ERROR":
        return <XCircle className="h-5 w-5 text-red-600" />;
      case "RUNNING":
      case "EXECUTING":
        return <Loader2 className="h-5 w-5 animate-spin text-blue-600" />;
      case "QUEUED":
        return <Clock className="h-5 w-5 text-amber-600" />;
      default:
        return <Play className="h-5 w-5 text-neutral-600" />;
    }
  };

  const getStatusColor = () => {
    switch (status) {
      case "COMPLETED":
        return "border-green-200 bg-green-50 dark:border-green-800 dark:bg-green-950/30";
      case "FAILED":
      case "ERROR":
        return "border-red-200 bg-red-50 dark:border-red-800 dark:bg-red-950/30";
      case "RUNNING":
      case "EXECUTING":
        return "border-blue-200 bg-blue-50 dark:border-blue-800 dark:bg-blue-950/30";
      case "QUEUED":
        return "border-amber-200 bg-amber-50 dark:border-amber-800 dark:bg-amber-950/30";
      default:
        return "border-neutral-200 bg-neutral-50 dark:border-neutral-800 dark:bg-neutral-950/30";
    }
  };

  const formatValue = (value: any): string => {
    if (value === null || value === undefined) return "null";
    if (typeof value === "object") {
      try {
        return JSON.stringify(value, null, 2);
      } catch {
        return String(value);
      }
    }
    return String(value);
  };

  return (
    <div
      className={cn(
        "my-4 overflow-hidden rounded-lg border transition-all duration-300",
        getStatusColor(),
        "animate-in fade-in-50 slide-in-from-bottom-2",
        className,
      )}
    >
      <div className="px-6 py-5">
        {/* Header */}
        <div className="mb-4 flex items-start justify-between">
          <div className="flex items-center gap-3">
            {getStatusIcon()}
            <div>
              <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100">
                {graphName}
              </h3>
              <p className="text-sm text-neutral-600 dark:text-neutral-400">
                Execution ID: {executionId.slice(0, 8)}...
              </p>
            </div>
          </div>
          <Link
            href={`/library/agents/${graphId}`}
            target="_blank"
            rel="noopener noreferrer"
          >
            <Button variant="outline" size="small">
              <ExternalLink className="mr-2 h-4 w-4" />
              Go to Run
            </Button>
          </Link>
        </div>

        {/* Input Data */}
        {inputs && Object.keys(inputs).length > 0 && (
          <div className="mb-4">
            <h4 className="mb-2 text-sm font-medium text-neutral-700 dark:text-neutral-300">
              Input Data:
            </h4>
            <div className="rounded-md bg-white/50 p-3 dark:bg-neutral-900/50">
              {Object.entries(inputs).map(([key, value]) => (
                <div key={key} className="mb-2 last:mb-0">
                  <span className="font-mono text-xs text-neutral-600 dark:text-neutral-400">
                    {key}:
                  </span>
                  <pre className="mt-1 overflow-x-auto rounded bg-neutral-100 p-2 text-xs text-neutral-800 dark:bg-neutral-800 dark:text-neutral-200">
                    {formatValue(value)}
                  </pre>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Output Data */}
        {outputs && Object.keys(outputs).length > 0 && (
          <div className="mb-4">
            <h4 className="mb-2 text-sm font-medium text-neutral-700 dark:text-neutral-300">
              Output Data:
            </h4>
            <div className="rounded-md bg-white/50 p-3 dark:bg-neutral-900/50">
              {Object.entries(outputs).map(([key, value]) => (
                <div key={key} className="mb-2 last:mb-0">
                  <span className="font-mono text-xs text-neutral-600 dark:text-neutral-400">
                    {key}:
                  </span>
                  <pre className="mt-1 overflow-x-auto rounded bg-neutral-100 p-2 text-xs text-neutral-800 dark:bg-neutral-800 dark:text-neutral-200">
                    {formatValue(value)}
                  </pre>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Error Message */}
        {error && (
          <div className="mb-4 rounded-md border border-red-200 bg-red-50 p-3 dark:border-red-800 dark:bg-red-950/50">
            <p className="text-sm text-red-700 dark:text-red-300">
              <strong>Error:</strong> {error}
            </p>
          </div>
        )}

        {/* Timeout Message */}
        {timeoutReached && (
          <div className="mb-4 rounded-md border border-amber-200 bg-amber-50 p-3 dark:border-amber-800 dark:bg-amber-950/50">
            <p className="text-sm text-amber-700 dark:text-amber-300">
              <strong>Note:</strong> Execution timed out after 30 seconds. The agent may still be running in the background.
            </p>
          </div>
        )}

        {/* Status Bar */}
        <div className="flex items-center justify-between text-xs">
          <span className="font-medium text-neutral-600 dark:text-neutral-400">
            Status: <span className="text-neutral-900 dark:text-neutral-100">{status}</span>
          </span>
          {endedAt && (
            <span className="text-neutral-500 dark:text-neutral-500">
              Ended: {new Date(endedAt).toLocaleString()}
            </span>
          )}
        </div>
      </div>
    </div>
  );
}