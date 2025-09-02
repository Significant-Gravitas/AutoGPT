"use client";

import React, { useState } from "react";
import { ChevronDown, ChevronUp, Wrench, Loader2, CheckCircle, XCircle } from "lucide-react";
import { cn } from "@/lib/utils";

interface ToolCallWidgetProps {
  toolName: string;
  parameters?: Record<string, any>;
  result?: string;
  status: "calling" | "executing" | "completed" | "error";
  error?: string;
  className?: string;
}

export function ToolCallWidget({
  toolName,
  parameters,
  result,
  status,
  error,
  className,
}: ToolCallWidgetProps) {
  const [isExpanded, setIsExpanded] = useState(true);

  const getStatusIcon = () => {
    switch (status) {
      case "calling":
        return <Loader2 className="h-4 w-4 animate-spin text-blue-500" />;
      case "executing":
        return <Loader2 className="h-4 w-4 animate-spin text-yellow-500" />;
      case "completed":
        return <CheckCircle className="h-4 w-4 text-green-500" />;
      case "error":
        return <XCircle className="h-4 w-4 text-red-500" />;
    }
  };

  const getStatusText = () => {
    switch (status) {
      case "calling":
        return "Preparing tool...";
      case "executing":
        return "Executing...";
      case "completed":
        return "Completed";
      case "error":
        return "Error";
    }
  };

  const getToolDisplayName = () => {
    const toolDisplayNames: Record<string, string> = {
      find_agent: "Search Marketplace",
      get_agent_details: "Get Agent Details",
      setup_agent: "Setup Agent",
    };
    return toolDisplayNames[toolName] || toolName;
  };

  return (
    <div
      className={cn(
        "my-4 overflow-hidden rounded-lg border",
        status === "error" ? "border-red-200 dark:border-red-800" : "border-neutral-200 dark:border-neutral-700",
        "bg-white dark:bg-neutral-900",
        "animate-in slide-in-from-top-2 duration-300",
        className
      )}
    >
      <div
        className={cn(
          "flex items-center justify-between px-4 py-3",
          "bg-gradient-to-r",
          status === "error" 
            ? "from-red-50 to-red-100 dark:from-red-900/20 dark:to-red-800/20"
            : "from-violet-50 to-purple-50 dark:from-violet-900/20 dark:to-purple-900/20"
        )}
      >
        <div className="flex items-center gap-3">
          <Wrench className="h-5 w-5 text-violet-600 dark:text-violet-400" />
          <span className="font-medium text-neutral-900 dark:text-neutral-100">
            {getToolDisplayName()}
          </span>
          <div className="flex items-center gap-2">
            {getStatusIcon()}
            <span className="text-sm text-neutral-600 dark:text-neutral-400">
              {getStatusText()}
            </span>
          </div>
        </div>
        
        <button
          onClick={() => setIsExpanded(!isExpanded)}
          className="rounded p-1 hover:bg-neutral-200/50 dark:hover:bg-neutral-700/50"
        >
          {isExpanded ? (
            <ChevronUp className="h-4 w-4 text-neutral-600 dark:text-neutral-400" />
          ) : (
            <ChevronDown className="h-4 w-4 text-neutral-600 dark:text-neutral-400" />
          )}
        </button>
      </div>

      {isExpanded && (
        <div className="px-4 py-3">
          {parameters && Object.keys(parameters).length > 0 && (
            <div className="mb-3">
              <div className="mb-2 text-xs font-medium text-neutral-600 dark:text-neutral-400">
                Parameters:
              </div>
              <div className="rounded-md bg-neutral-50 dark:bg-neutral-800 p-3">
                <pre className="text-xs text-neutral-700 dark:text-neutral-300">
                  {JSON.stringify(parameters, null, 2)}
                </pre>
              </div>
            </div>
          )}

          {result && status === "completed" && (
            <div>
              <div className="mb-2 text-xs font-medium text-neutral-600 dark:text-neutral-400">
                Result:
              </div>
              <div className="rounded-md bg-green-50 dark:bg-green-900/20 p-3">
                <pre className="whitespace-pre-wrap text-xs text-green-800 dark:text-green-200">
                  {result}
                </pre>
              </div>
            </div>
          )}

          {error && status === "error" && (
            <div>
              <div className="mb-2 text-xs font-medium text-red-600 dark:text-red-400">
                Error:
              </div>
              <div className="rounded-md bg-red-50 dark:bg-red-900/20 p-3">
                <p className="text-sm text-red-800 dark:text-red-200">{error}</p>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}