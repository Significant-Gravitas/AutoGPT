"use client";

import { useState } from "react";
import { Button } from "@/components/atoms/Button/Button";
import { Card } from "@/components/atoms/Card/Card";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { useDiagnosticsContent } from "./useDiagnosticsContent";
import { ExecutionsTable } from "./ExecutionsTable";
import { SchedulesTable } from "./SchedulesTable";
import { Refresh01Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";

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
    "all" | "orphaned" | "failed" | "long-running" | "stuck-queued" | "invalid"
  >("all");

  if (isLoading && !executionData && !agentData) {
    return (
      <div className="flex h-64 items-center justify-center">
        <div className="text-center">
          <Icon
            icon={Refresh01Icon}
            className="mx-auto h-8 w-8 animate-spin text-zinc-400"
          />
          <Text variant="large" tone="muted" className="mt-2">
            Loading diagnostics...
          </Text>
        </div>
      </div>
    );
  }

  if (isError) {
    return (
      <ErrorCard
        httpError={error as { status?: number; message?: string }}
        onRetry={refresh}
        context="diagnostics"
      />
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <Text variant="h3" as="h1">
            System Diagnostics
          </Text>
          <Text variant="large" tone="muted">
            Monitor execution and agent system health
          </Text>
        </div>
        <Button
          onClick={refresh}
          disabled={isLoading}
          variant="outline"
          size="md"
        >
          <Icon
            icon={Refresh01Icon}
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
                  <div className="flex flex-col space-y-1.5 p-6 pb-3">
                    <Text
                      variant="large-semibold"
                      as="h3"
                      className="text-orange-800"
                    >
                      Orphaned Executions
                    </Text>
                  </div>
                  <div className="p-6 pt-0">
                    <Text variant="h3" as="p" className="text-orange-900">
                      {executionData.orphaned_running +
                        executionData.orphaned_queued}
                    </Text>
                    <Text variant="body" className="text-orange-700">
                      {executionData.orphaned_running} running,{" "}
                      {executionData.orphaned_queued} queued ({">"}24h old)
                    </Text>
                    <Text variant="small" className="mt-2 text-orange-600">
                      Click to view →
                    </Text>
                  </div>
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
                  <div className="flex flex-col space-y-1.5 p-6 pb-3">
                    <Text
                      variant="large-semibold"
                      as="h3"
                      className="text-red-800"
                    >
                      Failed Executions (24h)
                    </Text>
                  </div>
                  <div className="p-6 pt-0">
                    <Text variant="h3" as="p" className="text-red-900">
                      {executionData.failed_count_24h}
                    </Text>
                    <Text variant="body" className="text-red-700">
                      {executionData.failed_count_1h} in last hour (
                      {executionData.failure_rate_24h.toFixed(1)}/hr rate)
                    </Text>
                    <Text variant="small" className="mt-2 text-red-600">
                      Click to view →
                    </Text>
                  </div>
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
                    <div className="flex flex-col space-y-1.5 p-6 pb-3">
                      <Text
                        variant="large-semibold"
                        as="h3"
                        className="text-yellow-800"
                      >
                        Long-Running Executions
                      </Text>
                    </div>
                    <div className="p-6 pt-0">
                      <Text variant="h3" as="p" className="text-yellow-900">
                        {executionData.stuck_running_24h}
                      </Text>
                      <Text variant="body" className="text-yellow-700">
                        Running {">"}24h (oldest:{" "}
                        {executionData.oldest_running_hours
                          ? `${Math.floor(executionData.oldest_running_hours)}h`
                          : "N/A"}
                        )
                      </Text>
                      <Text variant="small" className="mt-2 text-yellow-600">
                        Click to view →
                      </Text>
                    </div>
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
                  <div className="flex flex-col space-y-1.5 p-6 pb-3">
                    <Text
                      variant="large-semibold"
                      as="h3"
                      className="text-purple-800"
                    >
                      Orphaned Schedules
                    </Text>
                  </div>
                  <div className="p-6 pt-0">
                    <Text variant="h3" as="p" className="text-purple-900">
                      {scheduleData.total_orphaned}
                    </Text>
                    <Text variant="body" className="text-purple-700">
                      {scheduleData.orphaned_deleted_graph > 0 &&
                        `${scheduleData.orphaned_deleted_graph} deleted graph, `}
                      {scheduleData.orphaned_no_library_access > 0 &&
                        `${scheduleData.orphaned_no_library_access} no access`}
                    </Text>
                    <Text variant="small" className="mt-2 text-purple-600">
                      Click to view schedules →
                    </Text>
                  </div>
                </Card>
              </div>
            )}

            {/* Invalid State Alert */}
            {(executionData.invalid_queued_with_start > 0 ||
              executionData.invalid_running_without_start > 0) && (
              <div
                className="cursor-pointer transition-all hover:scale-105"
                onClick={() => setActiveTab("invalid")}
              >
                <Card className="border-pink-300 bg-pink-50">
                  <div className="flex flex-col space-y-1.5 p-6 pb-3">
                    <Text
                      variant="large-semibold"
                      as="h3"
                      className="text-pink-800"
                    >
                      Invalid States (Data Corruption)
                    </Text>
                  </div>
                  <div className="p-6 pt-0">
                    <Text variant="h3" as="p" className="text-pink-900">
                      {executionData.invalid_queued_with_start +
                        executionData.invalid_running_without_start}
                    </Text>
                    <Text variant="body" className="text-pink-700">
                      Requires manual investigation
                    </Text>
                    <Text variant="small" className="mt-2 text-pink-600">
                      Click to view (read-only) →
                    </Text>
                  </div>
                </Card>
              </div>
            )}
          </>
        )}
      </div>

      <div className="grid gap-6 md:grid-cols-3">
        <Card>
          <div className="flex flex-col space-y-1.5 p-6">
            <Text variant="large-semibold" as="h3">
              Execution Queue Status
            </Text>
            <Text variant="body" tone="muted">
              Current execution and queue metrics
            </Text>
          </div>
          <div className="p-6 pt-0">
            {executionData ? (
              <div className="space-y-4">
                <div className="flex items-center justify-between rounded-lg border p-4">
                  <div>
                    <Text variant="body-medium" tone="muted">
                      Running Executions
                    </Text>
                    <Text variant="h3" as="p">
                      {executionData.running_executions}
                    </Text>
                  </div>
                  <div className="flex h-12 w-12 items-center justify-center rounded-full bg-green-100">
                    <div className="h-6 w-6 rounded-full bg-green-500"></div>
                  </div>
                </div>

                <div className="flex items-center justify-between rounded-lg border p-4">
                  <div>
                    <Text variant="body-medium" tone="muted">
                      Queued in Database
                    </Text>
                    <Text variant="h3" as="p">
                      {executionData.queued_executions_db}
                    </Text>
                    {executionData.stuck_queued_1h > 0 && (
                      <Text variant="small" className="text-orange-600">
                        {executionData.stuck_queued_1h} stuck {">"}1h
                      </Text>
                    )}
                  </div>
                  <div className="flex h-12 w-12 items-center justify-center rounded-full bg-blue-100">
                    <div className="h-6 w-6 rounded-full bg-blue-500"></div>
                  </div>
                </div>

                <div className="flex items-center justify-between rounded-lg border p-4">
                  <div>
                    <Text variant="body-medium" tone="muted">
                      Queued in RabbitMQ
                    </Text>
                    <Text variant="h3" as="p">
                      {executionData.queued_executions_rabbitmq === -1 ? (
                        <span className="text-xl text-red-500">Error</span>
                      ) : (
                        executionData.queued_executions_rabbitmq
                      )}
                    </Text>
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

                <div className="text-xs text-zinc-400">
                  Last updated:{" "}
                  {new Date(executionData.timestamp).toLocaleString()}
                </div>
              </div>
            ) : (
              <Text variant="large" tone="muted">
                No data available
              </Text>
            )}
          </div>
        </Card>

        <Card>
          <div className="flex flex-col space-y-1.5 p-6">
            <Text variant="large-semibold" as="h3">
              System Throughput
            </Text>
            <Text variant="body" tone="muted">
              Execution completion and processing rates
            </Text>
          </div>
          <div className="p-6 pt-0">
            {executionData ? (
              <div className="space-y-4">
                <div className="flex items-center justify-between rounded-lg border p-4">
                  <div>
                    <Text variant="body-medium" tone="muted">
                      Completed (24h)
                    </Text>
                    <Text variant="h3" as="p">
                      {executionData.completed_24h}
                    </Text>
                    <Text variant="small" tone="secondary">
                      {executionData.completed_1h} in last hour
                    </Text>
                  </div>
                  <div className="flex h-12 w-12 items-center justify-center rounded-full bg-green-100">
                    <div className="h-6 w-6 rounded-full bg-green-500"></div>
                  </div>
                </div>

                <div className="flex items-center justify-between rounded-lg border p-4">
                  <div>
                    <Text variant="body-medium" tone="muted">
                      Throughput Rate
                    </Text>
                    <Text variant="h3" as="p">
                      {executionData.throughput_per_hour.toFixed(1)}
                    </Text>
                    <Text variant="small" tone="secondary">
                      completions per hour
                    </Text>
                  </div>
                  <div className="flex h-12 w-12 items-center justify-center rounded-full bg-blue-100">
                    <div className="h-6 w-6 rounded-full bg-blue-500"></div>
                  </div>
                </div>

                <div className="flex items-center justify-between rounded-lg border p-4">
                  <div>
                    <Text variant="body-medium" tone="muted">
                      Cancel Queue Depth
                    </Text>
                    <Text variant="h3" as="p">
                      {executionData.cancel_queue_depth === -1 ? (
                        <span className="text-xl text-red-500">Error</span>
                      ) : (
                        executionData.cancel_queue_depth
                      )}
                    </Text>
                  </div>
                  <div className="flex h-12 w-12 items-center justify-center rounded-full bg-purple-100">
                    <div className="h-6 w-6 rounded-full bg-purple-500"></div>
                  </div>
                </div>

                <div className="text-xs text-zinc-400">
                  Last updated:{" "}
                  {new Date(executionData.timestamp).toLocaleString()}
                </div>
              </div>
            ) : (
              <Text variant="large" tone="muted">
                No data available
              </Text>
            )}
          </div>
        </Card>

        <Card>
          <div className="flex flex-col space-y-1.5 p-6">
            <Text variant="large-semibold" as="h3">
              Schedules
            </Text>
            <Text variant="body" tone="muted">
              Scheduled agent executions and health
            </Text>
          </div>
          <div className="p-6 pt-0">
            {scheduleData ? (
              <div className="space-y-4">
                <div className="flex items-center justify-between rounded-lg border p-4">
                  <div>
                    <Text variant="body-medium" tone="muted">
                      User Schedules
                    </Text>
                    <Text variant="h3" as="p">
                      {scheduleData.user_schedules}
                    </Text>
                    {scheduleData.total_orphaned > 0 && (
                      <Text variant="small" className="text-orange-600">
                        {scheduleData.total_orphaned} orphaned
                      </Text>
                    )}
                  </div>
                  <div className="flex h-12 w-12 items-center justify-center rounded-full bg-purple-100">
                    <div className="h-6 w-6 rounded-full bg-purple-500"></div>
                  </div>
                </div>

                <div className="flex items-center justify-between rounded-lg border p-4">
                  <div>
                    <Text variant="body-medium" tone="muted">
                      Upcoming Runs (1h)
                    </Text>
                    <Text variant="h3" as="p">
                      {scheduleData.total_runs_next_hour}
                    </Text>
                    <Text variant="small" tone="secondary">
                      from {scheduleData.schedules_next_hour} schedule
                      {scheduleData.schedules_next_hour !== 1 ? "s" : ""}
                    </Text>
                  </div>
                  <div className="flex h-12 w-12 items-center justify-center rounded-full bg-blue-100">
                    <div className="h-6 w-6 rounded-full bg-blue-500"></div>
                  </div>
                </div>

                <div className="flex items-center justify-between rounded-lg border p-4">
                  <div>
                    <Text variant="body-medium" tone="muted">
                      Upcoming Runs (24h)
                    </Text>
                    <Text variant="h3" as="p">
                      {scheduleData.total_runs_next_24h}
                    </Text>
                    <Text variant="small" tone="secondary">
                      from {scheduleData.schedules_next_24h} schedule
                      {scheduleData.schedules_next_24h !== 1 ? "s" : ""}
                    </Text>
                  </div>
                  <div className="flex h-12 w-12 items-center justify-center rounded-full bg-green-100">
                    <div className="h-6 w-6 rounded-full bg-green-500"></div>
                  </div>
                </div>

                <div className="text-xs text-zinc-400">
                  Last updated:{" "}
                  {new Date(scheduleData.timestamp).toLocaleString()}
                </div>
              </div>
            ) : (
              <Text variant="large" tone="muted">
                No data available
              </Text>
            )}
          </div>
        </Card>
      </div>

      <Card>
        <div className="flex flex-col space-y-1.5 p-6">
          <Text variant="large-semibold" as="h3">
            Diagnostic Information
          </Text>
          <Text variant="body" tone="muted">
            Understanding metrics and tabs for on-call diagnostics
          </Text>
        </div>
        <div className="p-6 pt-0">
          <div className="space-y-3 text-sm">
            <div>
              <Text variant="body-medium" className="text-orange-700">
                🟠 Orphaned Executions:
              </Text>
              <Text variant="body" tone="secondary">
                Executions {">"}24h old in database but not actually running in
                executor. Usually from executor restarts/crashes. Safe to
                cleanup (marks as FAILED in DB).
              </Text>
            </div>
            <div>
              <Text variant="body-medium" className="text-blue-700">
                🔵 Stuck Queued Executions:
              </Text>
              <Text variant="body" tone="secondary">
                QUEUED {">"}1h but never started. Not in RabbitMQ queue. Can
                cleanup (safe) or requeue (⚠️ costs credits - only if temporary
                issue like RabbitMQ purge).
              </Text>
            </div>
            <div>
              <Text variant="body-medium" className="text-yellow-700">
                🟡 Long-Running Executions:
              </Text>
              <Text variant="body" tone="secondary">
                RUNNING status {">"}24h. May be legitimately long jobs or stuck.
                Review before stopping. Sends cancel signal to executor.
              </Text>
            </div>
            <div>
              <Text variant="body-medium" className="text-red-700">
                🔴 Failed Executions:
              </Text>
              <Text variant="body" tone="secondary">
                Executions that failed in last 24h. View error messages to
                identify patterns. Spike in failures indicates system issues.
              </Text>
            </div>
            <div>
              <Text variant="body-medium" className="text-pink-700">
                🩷 Invalid States (Data Corruption):
              </Text>
              <Text variant="body" tone="secondary">
                Executions in impossible states (QUEUED with startedAt, RUNNING
                without startedAt). Indicates DB corruption, race conditions, or
                crashes. Each requires manual investigation - no bulk actions
                provided.
              </Text>
            </div>
            <div>
              <Text variant="body-medium">Throughput Metrics:</Text>
              <Text variant="body" tone="secondary">
                Completions per hour shows system productivity. Declining
                throughput indicates performance degradation or executor issues.
              </Text>
            </div>
            <div>
              <Text variant="body-medium">Queue Health:</Text>
              <Text variant="body" tone="secondary">
                RabbitMQ depths should be low ({"<"}100). High queues indicate
                executor can&apos;t keep up. Cancel queue backlog indicates
                executor processing issues.
              </Text>
            </div>
          </div>
        </div>
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
                invalid_queued_with_start:
                  executionData.invalid_queued_with_start,
                invalid_running_without_start:
                  executionData.invalid_running_without_start,
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
