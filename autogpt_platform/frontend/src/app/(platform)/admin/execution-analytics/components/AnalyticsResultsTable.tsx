"use client";

import React, { useState } from "react";
import { Button } from "@/components/__legacy__/ui/button";
import { Badge } from "@/components/__legacy__/ui/badge";
import { DownloadIcon, EyeIcon } from "@phosphor-icons/react";

interface ExecutionAnalyticsResult {
  agent_id: string;
  version_id: number;
  user_id: string;
  exec_id: string;
  summary_text?: string;
  score?: number;
  status: "success" | "failed" | "skipped";
  error_message?: string;
}

interface ExecutionAnalyticsResponse {
  total_executions: number;
  processed_executions: number;
  successful_analytics: number;
  failed_analytics: number;
  skipped_executions: number;
  results: ExecutionAnalyticsResult[];
}

interface Props {
  results: ExecutionAnalyticsResponse;
}

export function AnalyticsResultsTable({ results }: Props) {
  const [expandedRows, setExpandedRows] = useState<Set<string>>(new Set());

  const toggleRowExpansion = (execId: string) => {
    const newExpanded = new Set(expandedRows);
    if (newExpanded.has(execId)) {
      newExpanded.delete(execId);
    } else {
      newExpanded.add(execId);
    }
    setExpandedRows(newExpanded);
  };

  const exportToCSV = () => {
    const headers = [
      "Agent ID",
      "Version",
      "User ID",
      "Execution ID",
      "Status",
      "Score",
      "Summary Text",
      "Error Message",
    ];

    const csvData = results.results.map((result) => [
      result.agent_id,
      result.version_id.toString(),
      result.user_id,
      result.exec_id,
      result.status,
      result.score?.toString() || "",
      `"${(result.summary_text || "").replace(/"/g, '""')}"`, // Escape quotes in summary
      `"${(result.error_message || "").replace(/"/g, '""')}"`, // Escape quotes in error
    ]);

    const csvContent = [
      headers.join(","),
      ...csvData.map((row) => row.join(",")),
    ].join("\n");

    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const link = document.createElement("a");
    const url = URL.createObjectURL(blob);

    link.setAttribute("href", url);
    link.setAttribute(
      "download",
      `execution-analytics-results-${new Date().toISOString().split("T")[0]}.csv`,
    );
    link.style.visibility = "hidden";

    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case "success":
        return <Badge variant="default">Success</Badge>;
      case "failed":
        return <Badge variant="destructive">Failed</Badge>;
      case "skipped":
        return <Badge variant="secondary">Skipped</Badge>;
      default:
        return <Badge variant="outline">{status}</Badge>;
    }
  };

  const getScoreDisplay = (score?: number) => {
    if (score === undefined || score === null) return "—";

    const percentage = Math.round(score * 100);
    let colorClass = "";

    if (score >= 0.8) colorClass = "text-green-600";
    else if (score >= 0.6) colorClass = "text-yellow-600";
    else if (score >= 0.4) colorClass = "text-orange-600";
    else colorClass = "text-red-600";

    return <span className={colorClass}>{percentage}%</span>;
  };

  return (
    <div className="space-y-4">
      {/* Summary Stats */}
      <div className="rounded-lg bg-gray-50 p-4">
        <h3 className="mb-3 text-lg font-semibold">Analytics Summary</h3>
        <div className="grid grid-cols-2 gap-4 text-sm md:grid-cols-5">
          <div>
            <span className="text-gray-600">Total Executions:</span>
            <div className="text-lg font-semibold">
              {results.total_executions}
            </div>
          </div>
          <div>
            <span className="text-gray-600">Processed:</span>
            <div className="text-lg font-semibold">
              {results.processed_executions}
            </div>
          </div>
          <div>
            <span className="text-gray-600">Successful:</span>
            <div className="text-lg font-semibold text-green-600">
              {results.successful_analytics}
            </div>
          </div>
          <div>
            <span className="text-gray-600">Failed:</span>
            <div className="text-lg font-semibold text-red-600">
              {results.failed_analytics}
            </div>
          </div>
          <div>
            <span className="text-gray-600">Skipped:</span>
            <div className="text-lg font-semibold text-gray-600">
              {results.skipped_executions}
            </div>
          </div>
        </div>
      </div>

      {/* Export Button */}
      <div className="flex justify-end">
        <Button
          variant="outline"
          onClick={exportToCSV}
          disabled={results.results.length === 0}
        >
          <DownloadIcon size={16} className="mr-2" />
          Export CSV
        </Button>
      </div>

      {/* Results Table */}
      {results.results.length > 0 ? (
        <div className="overflow-hidden rounded-lg border">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-4 py-3 text-left text-sm font-medium text-gray-600">
                    Agent ID
                  </th>
                  <th className="px-4 py-3 text-left text-sm font-medium text-gray-600">
                    Version
                  </th>
                  <th className="px-4 py-3 text-left text-sm font-medium text-gray-600">
                    User ID
                  </th>
                  <th className="px-4 py-3 text-left text-sm font-medium text-gray-600">
                    Execution ID
                  </th>
                  <th className="px-4 py-3 text-left text-sm font-medium text-gray-600">
                    Status
                  </th>
                  <th className="px-4 py-3 text-left text-sm font-medium text-gray-600">
                    Score
                  </th>
                  <th className="px-4 py-3 text-left text-sm font-medium text-gray-600">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200">
                {results.results.map((result) => (
                  <React.Fragment key={result.exec_id}>
                    <tr className="hover:bg-gray-50">
                      <td className="px-4 py-3 font-mono text-sm">
                        {result.agent_id.length > 20
                          ? `${result.agent_id.substring(0, 20)}...`
                          : result.agent_id}
                      </td>
                      <td className="px-4 py-3 text-sm">{result.version_id}</td>
                      <td className="px-4 py-3 font-mono text-sm">
                        {result.user_id.length > 20
                          ? `${result.user_id.substring(0, 20)}...`
                          : result.user_id}
                      </td>
                      <td className="px-4 py-3 font-mono text-sm">
                        {result.exec_id.length > 20
                          ? `${result.exec_id.substring(0, 20)}...`
                          : result.exec_id}
                      </td>
                      <td className="px-4 py-3 text-sm">
                        {getStatusBadge(result.status)}
                      </td>
                      <td className="px-4 py-3 text-sm">
                        {getScoreDisplay(result.score)}
                      </td>
                      <td className="px-4 py-3 text-sm">
                        {(result.summary_text || result.error_message) && (
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={() => toggleRowExpansion(result.exec_id)}
                          >
                            <EyeIcon size={16} />
                          </Button>
                        )}
                      </td>
                    </tr>

                    {expandedRows.has(result.exec_id) && (
                      <tr>
                        <td colSpan={7} className="bg-gray-50 px-4 py-3">
                          <div className="space-y-3">
                            {result.summary_text && (
                              <div>
                                <h4 className="mb-1 text-sm font-medium text-gray-700">
                                  Summary:
                                </h4>
                                <p className="text-sm leading-relaxed text-gray-600">
                                  {result.summary_text}
                                </p>
                              </div>
                            )}

                            {result.error_message && (
                              <div>
                                <h4 className="mb-1 text-sm font-medium text-red-700">
                                  Error:
                                </h4>
                                <p className="text-sm leading-relaxed text-red-600">
                                  {result.error_message}
                                </p>
                              </div>
                            )}
                          </div>
                        </td>
                      </tr>
                    )}
                  </React.Fragment>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ) : (
        <div className="py-8 text-center text-gray-500">
          No executions were processed.
        </div>
      )}
    </div>
  );
}
