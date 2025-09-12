"use client";

import React, { useState } from "react";
import {
  ChevronDown,
  ChevronUp,
  Wrench,
  Loader2,
  CheckCircle,
  XCircle,
} from "lucide-react";
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
  const [isExpanded, setIsExpanded] = useState(false);

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
      find_agent: "🔍 Search Marketplace",
      get_agent_details: "📋 Get Agent Details",
      check_credentials: "🔑 Check Credentials",
      setup_agent: "⚙️ Setup Agent",
      run_agent: "▶️ Run Agent",
      get_required_setup_info: "📝 Get Setup Requirements",
    };
    return toolDisplayNames[toolName] || toolName;
  };

  return (
    <div
      className={cn(
        "overflow-hidden rounded-lg border transition-all duration-200",
        status === "error"
          ? "border-red-200 dark:border-red-800"
          : "border-neutral-200 dark:border-neutral-700",
        "bg-white dark:bg-neutral-900",
        "animate-in fade-in-50 slide-in-from-top-1",
        className,
      )}
    >
      <div
        className={cn(
          "flex items-center justify-between px-3 py-2",
          "bg-gradient-to-r",
          status === "error"
            ? "from-red-50 to-red-100 dark:from-red-900/20 dark:to-red-800/20"
            : "from-neutral-50 to-neutral-100 dark:from-neutral-800/20 dark:to-neutral-700/20",
        )}
      >
        <div className="flex items-center gap-2">
          <Wrench className="h-4 w-4 text-neutral-500 dark:text-neutral-400" />
          <span className="text-sm font-medium text-neutral-700 dark:text-neutral-300">
            {getToolDisplayName()}
          </span>
          <div className="ml-2 flex items-center gap-1.5">
            {getStatusIcon()}
            <span className="text-xs text-neutral-500 dark:text-neutral-400">
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
              <div className="rounded-md bg-neutral-50 p-3 dark:bg-neutral-800">
                <pre className="text-xs text-neutral-700 dark:text-neutral-300">
                  {JSON.stringify(parameters, null, 2)}
                </pre>
              </div>
            </div>
          )}

          {result &&
            status === "completed" &&
            (() => {
              // Check if result is agent carousel data - if so, don't display raw JSON
              try {
                const parsed = JSON.parse(result);
                if (parsed.type === "agent_carousel") {
                  return (
                    <div className="text-xs text-neutral-600 dark:text-neutral-400">
                      Found {parsed.count} agents matching &ldquo;{parsed.query}
                      &rdquo;
                    </div>
                  );
                }
              } catch {}

              // Display regular result
              return (
                <div>
                  <div className="mb-2 text-xs font-medium text-neutral-600 dark:text-neutral-400">
                    Result:
                  </div>
                  <div className="rounded-md bg-green-50 p-3 dark:bg-green-900/20">
                    <pre className="whitespace-pre-wrap text-xs text-green-800 dark:text-green-200">
                      {result}
                    </pre>
                  </div>
                </div>
              );
            })()}

          {error && status === "error" && (
            <div>
              <div className="mb-2 text-xs font-medium text-red-600 dark:text-red-400">
                Error:
              </div>
              <div className="rounded-md bg-red-50 p-3 dark:bg-red-900/20">
                <p className="text-sm text-red-800 dark:text-red-200">
                  {error}
                </p>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
