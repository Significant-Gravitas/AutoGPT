import { ApiError } from "@/lib/autogpt-server-api/helpers";
import Link from "next/link";
import React from "react";

type ValidationErrorDetail = {
  type: string;
  message?: string;
  node_errors?: Record<string, Record<string, string>>;
};

type AgentInfo = {
  graph_id: string;
  graph_version: number;
};

export function formatValidationError(
  error: any,
  agentInfo?: AgentInfo,
): string | React.ReactNode {
  if (
    !(error instanceof ApiError) ||
    !error.isGraphValidationError() ||
    !error.response?.detail
  ) {
    return error.message || "An unexpected error occurred.";
  }

  const detail: ValidationErrorDetail = error.response.detail;

  // Format validation errors nicely
  if (detail.type === "validation_error" && detail.node_errors) {
    const nodeErrors = detail.node_errors;
    const errorItems: React.ReactNode[] = [];

    // Collect all field errors
    Object.entries(nodeErrors).forEach(([nodeId, fields]) => {
      if (fields && typeof fields === "object") {
        Object.entries(fields).forEach(([fieldName, fieldError]) => {
          errorItems.push(
            <div key={`${nodeId}-${fieldName}`} className="mt-1">
              <span className="font-medium">{fieldName}:</span>{" "}
              {String(fieldError)}
            </div>,
          );
        });
      }
    });

    if (errorItems.length > 0) {
      return (
        <div className="space-y-1">
          <div className="font-medium text-white">
            {detail.message || "Validation failed"}
          </div>
          <div className="mt-2 space-y-1 text-xs">{errorItems}</div>
          {agentInfo && (
            <div className="mt-3 text-xs">
              Check the agent graph and try to run from there for further
              details.{" "}
              <Link
                href={`/build?flowID=${agentInfo.graph_id}&flowVersion=${agentInfo.graph_version}`}
                target="_blank"
                rel="noopener noreferrer"
                className="cursor-pointer underline hover:no-underline"
              >
                Open in builder
              </Link>
            </div>
          )}
        </div>
      );
    } else {
      return detail.message || "Validation failed";
    }
  }

  return detail.message || error.message || "An unexpected error occurred.";
}

export function showExecutionErrorToast(
  toast: (options: {
    title: string;
    description: string | React.ReactNode;
    variant: "destructive";
    duration: number;
    dismissable: boolean;
  }) => void,
  error: any,
  agentInfo?: AgentInfo,
) {
  const errorMessage = formatValidationError(error, agentInfo);

  toast({
    title: "Failed to execute agent",
    description: errorMessage,
    variant: "destructive",
    duration: 10000, // 10 seconds - long enough to read and close
    dismissable: true,
  });
}
