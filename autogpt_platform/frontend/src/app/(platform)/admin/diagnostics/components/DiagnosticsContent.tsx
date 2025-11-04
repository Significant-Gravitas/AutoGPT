"use client";

import { useState } from "react";
import { Button } from "@/components/atoms/Button/Button";
import { Card } from "@/components/atoms/Card/Card";
import {
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/__legacy__/ui/card";
import { ArrowClockwise } from "@phosphor-icons/react";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { useDiagnosticsContent } from "./useDiagnosticsContent";
import { ExecutionsTable } from "./ExecutionsTable";
import { SchedulesTable } from "./SchedulesTable";

export function DiagnosticsContent() {
  const {
    executionData,
    agentData,
    scheduleData,
    isLoading,
    isError,
    error,
    refresh,
  } = useDiagnosticsContent();

  const [activeTab, setActiveTab] = useState<
    "all" | "orphaned" | "failed" | "long-running" | "stuck-queued"
  >("all");

  if (isLoading && !executionData && !agentData) {
    return (
      <div className="flex h-64 items-center justify-center">
        <div className="text-center">
          <ArrowClockwise className="mx-auto h-8 w-8 animate-spin text-gray-400" />
          <p className="mt-2 text-gray-500">Loading diagnostics...</p>
        </div>
      </div>
    );
  }

  if (isError) {
    return (
      <ErrorCard
        httpError={error as any}
        onRetry={refresh}
        context="diagnostics"
      />
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">System Diagnostics</h1>
          <p className="text-gray-500">
            Monitor execution and agent system health
          </p>
        </div>
        <Button
          onClick={refresh}
          disabled={isLoading}
          variant="outline"
          size="small"
        >
          <ArrowClockwise
            className={`mr-2 h-4 w-4 ${isLoading ? "animate-spin" : ""}`}
          />
          Refresh
        </Button>
      </div>

      {/* Alert Cards for Critical Issues */}
      <div className="grid gap-4 md:grid-cols-3">
        {executionData && (
          <>
            {/* Orphaned Executions Alert */}
            {(executionData.orphaned_running > 0 ||
              executionData.orphaned_queued > 0) && (
              <div
                className="cursor-pointer transition-all hover:scale-105"
                onClick={() => setActiveTab("orphaned")}
              >
                <Card className="border-orange-300 bg-orange-50">
                  <CardHeader className="pb-3">
                    <CardTitle className="text-orange-800">
                      Orphaned Executions
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-3xl font-bold text-orange-900">
                      {executionData.orphaned_running +
                        executionData.orphaned_queued}
                    </p>
                    <p className="text-sm text-orange-700">
                      {executionData.orphaned_running} running,{" "}
                      {executionData.orphaned_queued} queued ({">"}24h old)
                    </p>
                    <p className="mt-2 text-xs text-orange-600">
                      Click to view →
                    </p>
                  </CardContent>
                </Card>
              </div>
            )}

            {/* Failed Executions Alert */}
            {executionData.failed_count_24h > 0 && (
              <div
                className="cursor-pointer transition-all hover:scale-105"
                onClick={() => setActiveTab("failed")}
              >
                <Card className="border-red-300 bg-red-50">
                  <CardHeader className="pb-3">
                    <CardTitle className="text-red-800">
                      Failed Executions (24h)
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-3xl font-bold text-red-900">
                      {executionData.failed_count_24h}
                    </p>
                    <p className="text-sm text-red-700">
                      {executionData.failed_count_1h} in last hour (
                      {executionData.failure_rate_24h.toFixed(1)}/hr rate)
                    </p>
                    <p className="mt-2 text-xs text-red-600">Click to view →</p>
                  </CardContent>
                </Card>
              </div>
            )}

            {/* Long-Running Alert */}
            {executionData.stuck_running_24h > 0 && (
              <>
                <div
                  className="cursor-pointer transition-all hover:scale-105"
                  onClick={() => setActiveTab("long-running")}
                >
                  <Card className="border-yellow-300 bg-yellow-50">
                    <CardHeader className="pb-3">
                      <CardTitle className="text-yellow-800">
                        Long-Running Executions
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <p className="text-3xl font-bold text-yellow-900">
                        {executionData.stuck_running_24h}
                      </p>
                      <p className="text-sm text-yellow-700">
                        Running {">"}24h (oldest:{" "}
                        {executionData.oldest_running_hours
                          ? `${Math.floor(executionData.oldest_running_hours)}h`
                          : "N/A"}
                        )
                      </p>
                      <p className="mt-2 text-xs text-yellow-600">
                        Click to view →
                      </p>
                    </CardContent>
                  </Card>
                </div>
              </>
            )}

            {/* Orphaned Schedules Alert */}
            {scheduleData && scheduleData.total_orphaned > 0 && (
              <div
                className="cursor-pointer transition-all hover:scale-105"
                onClick={() => setActiveTab("all")}
              >
                <Card className="border-purple-300 bg-purple-50">
                  <CardHeader className="pb-3">
                    <CardTitle className="text-purple-800">
                      Orphaned Schedules
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-3xl font-bold text-purple-900">
                      {scheduleData.total_orphaned}
                    </p>
                    <p className="text-sm text-purple-700">
                      {scheduleData.orphaned_deleted_graph > 0 &&
                        `${scheduleData.orphaned_deleted_graph} deleted graph, `}
                      {scheduleData.orphaned_no_library_access > 0 &&
                        `${scheduleData.orphaned_no_library_access} no access`}
                    </p>
                    <p className="mt-2 text-xs text-purple-600">
                      Click to view schedules →
                    </p>
                  </CardContent>
                </Card>
              </div>
            )}
          </>
        )}
      </div>

      <div className="grid gap-6 md:grid-cols-3">
        <Card>
          <CardHeader>
            <CardTitle>Execution Queue Status</CardTitle>
            <CardDescription>
              Current execution and queue metrics
            </CardDescription>
          </CardHeader>
          <CardContent>
            {executionData ? (
              <div className="space-y-4">
                <div className="flex items-center justify-between rounded-lg border p-4">
                  <div>
                    <p className="text-sm font-medium text-gray-500">
                      Running Executions
                    </p>
                    <p className="text-3xl font-bold">
                      {executionData.running_executions}
                    </p>
                  </div>
                  <div className="flex h-12 w-12 items-center justify-center rounded-full bg-green-100">
                    <div className="h-6 w-6 rounded-full bg-green-500"></div>
                  </div>
                </div>

                <div className="flex items-center justify-between rounded-lg border p-4">
                  <div>
                    <p className="text-sm font-medium text-gray-500">
                      Queued in Database
                    </p>
                    <p className="text-3xl font-bold">
                      {executionData.queued_executions_db}
                    </p>
                    {executionData.stuck_queued_1h > 0 && (
                      <p className="text-xs text-orange-600">
                        {executionData.stuck_queued_1h} stuck {">"}1h
                      </p>
                    )}
                  </div>
                  <div className="flex h-12 w-12 items-center justify-center rounded-full bg-blue-100">
                    <div className="h-6 w-6 rounded-full bg-blue-500"></div>
                  </div>
                </div>

                <div className="flex items-center justify-between rounded-lg border p-4">
                  <div>
                    <p className="text-sm font-medium text-gray-500">
                      Queued in RabbitMQ
                    </p>
                    <p className="text-3xl font-bold">
                      {executionData.queued_executions_rabbitmq === -1 ? (
                        <span className="text-xl text-red-500">Error</span>
                      ) : (
                        executionData.queued_executions_rabbitmq
                      )}
                    </p>
                  </div>
                  <div
                    className={`flex h-12 w-12 items-center justify-center rounded-full ${
                      executionData.queued_executions_rabbitmq === -1
                        ? "bg-red-100"
                        : "bg-yellow-100"
                    }`}
                  >
                    <div
                      className={`h-6 w-6 rounded-full ${
                        executionData.queued_executions_rabbitmq === -1
                          ? "bg-red-500"
                          : "bg-yellow-500"
                      }`}
                    ></div>
                  </div>
                </div>

                <div className="text-xs text-gray-400">
                  Last updated:{" "}
                  {new Date(executionData.timestamp).toLocaleString()}
                </div>
              </div>
            ) : (
              <p className="text-gray-500">No data available</p>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>System Throughput</CardTitle>
            <CardDescription>
              Execution completion and processing rates
            </CardDescription>
          </CardHeader>
          <CardContent>
            {executionData ? (
              <div className="space-y-4">
                <div className="flex items-center justify-between rounded-lg border p-4">
                  <div>
                    <p className="text-sm font-medium text-gray-500">
                      Completed (24h)
                    </p>
                    <p className="text-3xl font-bold">
                      {executionData.completed_24h}
                    </p>
                    <p className="text-xs text-gray-600">
                      {executionData.completed_1h} in last hour
                    </p>
                  </div>
                  <div className="flex h-12 w-12 items-center justify-center rounded-full bg-green-100">
                    <div className="h-6 w-6 rounded-full bg-green-500"></div>
                  </div>
                </div>

                <div className="flex items-center justify-between rounded-lg border p-4">
                  <div>
                    <p className="text-sm font-medium text-gray-500">
                      Throughput Rate
                    </p>
                    <p className="text-3xl font-bold">
                      {executionData.throughput_per_hour.toFixed(1)}
                    </p>
                    <p className="text-xs text-gray-600">
                      completions per hour
                    </p>
                  </div>
                  <div className="flex h-12 w-12 items-center justify-center rounded-full bg-blue-100">
                    <div className="h-6 w-6 rounded-full bg-blue-500"></div>
                  </div>
                </div>

                <div className="flex items-center justify-between rounded-lg border p-4">
                  <div>
                    <p className="text-sm font-medium text-gray-500">
                      Cancel Queue Depth
                    </p>
                    <p className="text-3xl font-bold">
                      {executionData.cancel_queue_depth === -1 ? (
                        <span className="text-xl text-red-500">Error</span>
                      ) : (
                        executionData.cancel_queue_depth
                      )}
                    </p>
                  </div>
                  <div className="flex h-12 w-12 items-center justify-center rounded-full bg-purple-100">
                    <div className="h-6 w-6 rounded-full bg-purple-500"></div>
                  </div>
                </div>

                <div className="text-xs text-gray-400">
                  Last updated:{" "}
                  {new Date(executionData.timestamp).toLocaleString()}
                </div>
              </div>
            ) : (
              <p className="text-gray-500">No data available</p>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Schedules</CardTitle>
            <CardDescription>
              Scheduled agent executions and health
            </CardDescription>
          </CardHeader>
          <CardContent>
            {scheduleData ? (
              <div className="space-y-4">
                <div className="flex items-center justify-between rounded-lg border p-4">
                  <div>
                    <p className="text-sm font-medium text-gray-500">
                      User Schedules
                    </p>
                    <p className="text-3xl font-bold">
                      {scheduleData.user_schedules}
                    </p>
                    {scheduleData.total_orphaned > 0 && (
                      <p className="text-xs text-orange-600">
                        {scheduleData.total_orphaned} orphaned
                      </p>
                    )}
                  </div>
                  <div className="flex h-12 w-12 items-center justify-center rounded-full bg-purple-100">
                    <div className="h-6 w-6 rounded-full bg-purple-500"></div>
                  </div>
                </div>

                <div className="flex items-center justify-between rounded-lg border p-4">
                  <div>
                    <p className="text-sm font-medium text-gray-500">
                      Next Hour
                    </p>
                    <p className="text-3xl font-bold">
                      {scheduleData.schedules_next_hour}
                    </p>
                    <p className="text-xs text-gray-600">
                      {scheduleData.schedules_next_24h} in next 24h
                    </p>
                  </div>
                  <div className="flex h-12 w-12 items-center justify-center rounded-full bg-blue-100">
                    <div className="h-6 w-6 rounded-full bg-blue-500"></div>
                  </div>
                </div>

                <div className="text-xs text-gray-400">
                  Last updated:{" "}
                  {new Date(scheduleData.timestamp).toLocaleString()}
                </div>
              </div>
            ) : (
              <p className="text-gray-500">No data available</p>
            )}
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Diagnostic Information</CardTitle>
          <CardDescription>
            Understanding metrics and tabs for on-call diagnostics
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-3 text-sm">
            <div>
              <p className="font-semibold text-orange-700">
                🟠 Orphaned Executions:
              </p>
              <p className="text-gray-600">
                Executions {">"}24h old in database but not actually running in
                executor. Usually from executor restarts/crashes. Safe to
                cleanup (marks as FAILED in DB).
              </p>
            </div>
            <div>
              <p className="font-semibold text-blue-700">
                🔵 Stuck Queued Executions:
              </p>
              <p className="text-gray-600">
                QUEUED {">"}1h but never started. Not in RabbitMQ queue. Can
                cleanup (safe) or requeue (⚠️ costs credits - only if temporary
                issue like RabbitMQ purge).
              </p>
            </div>
            <div>
              <p className="font-semibold text-yellow-700">
                🟡 Long-Running Executions:
              </p>
              <p className="text-gray-600">
                RUNNING status {">"}24h. May be legitimately long jobs or stuck.
                Review before stopping. Sends cancel signal to executor.
              </p>
            </div>
            <div>
              <p className="font-semibold text-red-700">
                🔴 Failed Executions:
              </p>
              <p className="text-gray-600">
                Executions that failed in last 24h. View error messages to
                identify patterns. Spike in failures indicates system issues.
              </p>
            </div>
            <div>
              <p className="font-semibold">Throughput Metrics:</p>
              <p className="text-gray-600">
                Completions per hour shows system productivity. Declining
                throughput indicates performance degradation or executor issues.
              </p>
            </div>
            <div>
              <p className="font-semibold">Queue Health:</p>
              <p className="text-gray-600">
                RabbitMQ depths should be low ({"<"}100). High queues indicate
                executor can&apos;t keep up. Cancel queue backlog indicates
                executor processing issues.
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Add Executions Table with tab counts */}
      <ExecutionsTable
        onRefresh={refresh}
        initialTab={activeTab}
        onTabChange={setActiveTab}
        diagnosticsData={
          executionData
            ? {
                orphaned_running: executionData.orphaned_running,
                orphaned_queued: executionData.orphaned_queued,
                failed_count_24h: executionData.failed_count_24h,
                stuck_running_24h: executionData.stuck_running_24h,
                stuck_queued_1h: executionData.stuck_queued_1h,
              }
            : undefined
        }
      />

      {/* Add Schedules Table */}
      <SchedulesTable
        onRefresh={refresh}
        diagnosticsData={
          scheduleData
            ? {
                total_orphaned: scheduleData.total_orphaned,
                user_schedules: scheduleData.user_schedules,
              }
            : undefined
        }
      />
    </div>
  );
}
