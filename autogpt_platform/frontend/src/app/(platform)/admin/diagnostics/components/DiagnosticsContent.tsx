"use client";

import { useState } from "react";
import { Button } from "@/components/atoms/Button/Button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/atoms/Card/Card";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { DiagnosticAlertCard } from "./DiagnosticAlertCard";
import { ErrorCard } from "@/components/molecules/ErrorCard/ErrorCard";
import { useDiagnosticsContent } from "./useDiagnosticsContent";
import { ExecutionsTable } from "./ExecutionsTable";
import { SchedulesTable } from "./SchedulesTable";
import { Refresh01Icon } from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

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
          <span className="mx-auto block h-8 w-8 animate-spin text-foreground-subtle">
            <Icon icon={Refresh01Icon} className="h-8 w-8" />
          </span>
          <p className="mt-2 text-foreground-tertiary">
            Loading diagnostics...
          </p>
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
          <h1 className="text-3xl font-bold">System Diagnostics</h1>
          <p className="text-foreground-tertiary">
            Monitor execution and agent system health
          </p>
        </div>
        <Button
          onClick={refresh}
          disabled={isLoading}
          variant="outline"
          size="small"
        >
          <span
            className={cn("mr-2 block h-4 w-4", isLoading && "animate-spin")}
          >
            <Icon icon={Refresh01Icon} className="h-4 w-4" />
          </span>
          Refresh
        </Button>
      </div>

      {/* Alert Cards for Critical Issues */}
      <div className="grid gap-4 md:grid-cols-3">
        {executionData && (
          <>
            {(executionData.orphaned_running > 0 ||
              executionData.orphaned_queued > 0) && (
              <DiagnosticAlertCard
                tone="orange"
                title="Orphaned Executions"
                value={
                  executionData.orphaned_running + executionData.orphaned_queued
                }
                detail={
                  <>
                    {executionData.orphaned_running} running,{" "}
                    {executionData.orphaned_queued} queued ({">"}24h old)
                  </>
                }
                hint="Click to view →"
                onClick={() => setActiveTab("orphaned")}
              />
            )}

            {executionData.failed_count_24h > 0 && (
              <DiagnosticAlertCard
                tone="error"
                title="Failed Executions (24h)"
                value={executionData.failed_count_24h}
                detail={
                  <>
                    {executionData.failed_count_1h} in last hour (
                    {executionData.failure_rate_24h.toFixed(1)}/hr rate)
                  </>
                }
                hint="Click to view →"
                onClick={() => setActiveTab("failed")}
              />
            )}

            {executionData.stuck_running_24h > 0 && (
              <DiagnosticAlertCard
                tone="yellow"
                title="Long-Running Executions"
                value={executionData.stuck_running_24h}
                detail={
                  <>
                    Running {">"}24h (oldest:{" "}
                    {executionData.oldest_running_hours
                      ? `${Math.floor(executionData.oldest_running_hours)}h`
                      : "N/A"}
                    )
                  </>
                }
                hint="Click to view →"
                onClick={() => setActiveTab("long-running")}
              />
            )}

            {scheduleData && scheduleData.total_orphaned > 0 && (
              <DiagnosticAlertCard
                tone="brand"
                title="Orphaned Schedules"
                value={scheduleData.total_orphaned}
                detail={
                  <>
                    {scheduleData.orphaned_deleted_graph > 0 &&
                      `${scheduleData.orphaned_deleted_graph} deleted graph, `}
                    {scheduleData.orphaned_no_library_access > 0 &&
                      `${scheduleData.orphaned_no_library_access} no access`}
                  </>
                }
                hint="Click to view schedules →"
                onClick={() => setActiveTab("all")}
              />
            )}

            {(executionData.invalid_queued_with_start > 0 ||
              executionData.invalid_running_without_start > 0) && (
              <DiagnosticAlertCard
                tone="pink"
                title="Invalid States (Data Corruption)"
                value={
                  executionData.invalid_queued_with_start +
                  executionData.invalid_running_without_start
                }
                detail="Requires manual investigation"
                hint="Click to view (read-only) →"
                onClick={() => setActiveTab("invalid")}
              />
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
                    <Text variant="body-medium-tight" tone="muted">
                      Running Executions
                    </Text>
                    <p className="text-3xl font-bold">
                      {executionData.running_executions}
                    </p>
                  </div>
                  <div className="flex h-12 w-12 items-center justify-center rounded-full bg-success/30">
                    <div className="h-6 w-6 rounded-full bg-success"></div>
                  </div>
                </div>

                <div className="flex items-center justify-between rounded-lg border p-4">
                  <div>
                    <Text variant="body-medium-tight" tone="muted">
                      Queued in Database
                    </Text>
                    <p className="text-3xl font-bold">
                      {executionData.queued_executions_db}
                    </p>
                    {executionData.stuck_queued_1h > 0 && (
                      // oxlint-disable-next-line design-system/no-raw-colors -- orange has no role; amber would change the hue (deferred)
                      <p className="text-xs text-orange-600">
                        {executionData.stuck_queued_1h} stuck {">"}1h
                      </p>
                    )}
                  </div>
                  <div className="flex h-12 w-12 items-center justify-center rounded-full bg-info/20">
                    <div className="h-6 w-6 rounded-full bg-info"></div>
                  </div>
                </div>

                <div className="flex items-center justify-between rounded-lg border p-4">
                  <div>
                    <Text variant="body-medium-tight" tone="muted">
                      Queued in RabbitMQ
                    </Text>
                    <p className="text-3xl font-bold">
                      {executionData.queued_executions_rabbitmq === -1 ? (
                        <span className="text-xl text-error">Error</span>
                      ) : (
                        executionData.queued_executions_rabbitmq
                      )}
                    </p>
                  </div>
                  <div
                    className={cn(
                      "flex h-12 w-12 items-center justify-center rounded-full",
                      executionData.queued_executions_rabbitmq === -1
                        ? "bg-error-surface"
                        : // oxlint-disable-next-line design-system/no-raw-colors -- yellow has no role; amber would change the hue (deferred)
                          "bg-yellow-100",
                    )}
                  >
                    <div
                      className={cn(
                        "h-6 w-6 rounded-full",
                        executionData.queued_executions_rabbitmq === -1
                          ? "bg-error"
                          : // oxlint-disable-next-line design-system/no-raw-colors -- yellow has no role; amber would change the hue (deferred)
                            "bg-yellow-500",
                      )}
                    ></div>
                  </div>
                </div>

                <div className="text-xs text-foreground-subtle">
                  Last updated:{" "}
                  {new Date(executionData.timestamp).toLocaleString()}
                </div>
              </div>
            ) : (
              <p className="text-foreground-tertiary">No data available</p>
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
                    <Text variant="body-medium-tight" tone="muted">
                      Completed (24h)
                    </Text>
                    <p className="text-3xl font-bold">
                      {executionData.completed_24h}
                    </p>
                    <Text variant="small-tight" tone="secondary">
                      {executionData.completed_1h} in last hour
                    </Text>
                  </div>
                  <div className="flex h-12 w-12 items-center justify-center rounded-full bg-success/30">
                    <div className="h-6 w-6 rounded-full bg-success"></div>
                  </div>
                </div>

                <div className="flex items-center justify-between rounded-lg border p-4">
                  <div>
                    <Text variant="body-medium-tight" tone="muted">
                      Throughput Rate
                    </Text>
                    <p className="text-3xl font-bold">
                      {executionData.throughput_per_hour.toFixed(1)}
                    </p>
                    <Text variant="small-tight" tone="secondary">
                      completions per hour
                    </Text>
                  </div>
                  <div className="flex h-12 w-12 items-center justify-center rounded-full bg-info/20">
                    <div className="h-6 w-6 rounded-full bg-info"></div>
                  </div>
                </div>

                <div className="flex items-center justify-between rounded-lg border p-4">
                  <div>
                    <Text variant="body-medium-tight" tone="muted">
                      Cancel Queue Depth
                    </Text>
                    <p className="text-3xl font-bold">
                      {executionData.cancel_queue_depth === -1 ? (
                        <span className="text-xl text-error">Error</span>
                      ) : (
                        executionData.cancel_queue_depth
                      )}
                    </p>
                  </div>
                  <div className="flex h-12 w-12 items-center justify-center rounded-full bg-accent-surface">
                    <div className="h-6 w-6 rounded-full bg-brand"></div>
                  </div>
                </div>

                <div className="text-xs text-foreground-subtle">
                  Last updated:{" "}
                  {new Date(executionData.timestamp).toLocaleString()}
                </div>
              </div>
            ) : (
              <p className="text-foreground-tertiary">No data available</p>
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
                    <Text variant="body-medium-tight" tone="muted">
                      User Schedules
                    </Text>
                    <p className="text-3xl font-bold">
                      {scheduleData.user_schedules}
                    </p>
                    {scheduleData.total_orphaned > 0 && (
                      // oxlint-disable-next-line design-system/no-raw-colors -- orange has no role; amber would change the hue (deferred)
                      <p className="text-xs text-orange-600">
                        {scheduleData.total_orphaned} orphaned
                      </p>
                    )}
                  </div>
                  <div className="flex h-12 w-12 items-center justify-center rounded-full bg-accent-surface">
                    <div className="h-6 w-6 rounded-full bg-brand"></div>
                  </div>
                </div>

                <div className="flex items-center justify-between rounded-lg border p-4">
                  <div>
                    <Text variant="body-medium-tight" tone="muted">
                      Upcoming Runs (1h)
                    </Text>
                    <p className="text-3xl font-bold">
                      {scheduleData.total_runs_next_hour}
                    </p>
                    <Text variant="small-tight" tone="secondary">
                      from {scheduleData.schedules_next_hour} schedule
                      {scheduleData.schedules_next_hour !== 1 ? "s" : ""}
                    </Text>
                  </div>
                  <div className="flex h-12 w-12 items-center justify-center rounded-full bg-info/20">
                    <div className="h-6 w-6 rounded-full bg-info"></div>
                  </div>
                </div>

                <div className="flex items-center justify-between rounded-lg border p-4">
                  <div>
                    <Text variant="body-medium-tight" tone="muted">
                      Upcoming Runs (24h)
                    </Text>
                    <p className="text-3xl font-bold">
                      {scheduleData.total_runs_next_24h}
                    </p>
                    <Text variant="small-tight" tone="secondary">
                      from {scheduleData.schedules_next_24h} schedule
                      {scheduleData.schedules_next_24h !== 1 ? "s" : ""}
                    </Text>
                  </div>
                  <div className="flex h-12 w-12 items-center justify-center rounded-full bg-success/30">
                    <div className="h-6 w-6 rounded-full bg-success"></div>
                  </div>
                </div>

                <div className="text-xs text-foreground-subtle">
                  Last updated:{" "}
                  {new Date(scheduleData.timestamp).toLocaleString()}
                </div>
              </div>
            ) : (
              <p className="text-foreground-tertiary">No data available</p>
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
              {/* oxlint-disable-next-line design-system/no-raw-colors -- orange has no role; amber would change the hue (deferred) */}
              <p className="font-semibold text-orange-700">
                🟠 Orphaned Executions:
              </p>
              <p className="text-foreground-secondary">
                Executions {">"}24h old in database but not actually running in
                executor. Usually from executor restarts/crashes. Safe to
                cleanup (marks as FAILED in DB).
              </p>
            </div>
            <div>
              <p className="font-semibold text-info-foreground">
                🔵 Stuck Queued Executions:
              </p>
              <p className="text-foreground-secondary">
                QUEUED {">"}1h but never started. Not in RabbitMQ queue. Can
                cleanup (safe) or requeue (⚠️ costs credits - only if temporary
                issue like RabbitMQ purge).
              </p>
            </div>
            <div>
              {/* oxlint-disable-next-line design-system/no-raw-colors -- yellow has no role; amber would change the hue (deferred) */}
              <p className="font-semibold text-yellow-700">
                🟡 Long-Running Executions:
              </p>
              <p className="text-foreground-secondary">
                RUNNING status {">"}24h. May be legitimately long jobs or stuck.
                Review before stopping. Sends cancel signal to executor.
              </p>
            </div>
            <div>
              <p className="font-semibold text-error-foreground">
                🔴 Failed Executions:
              </p>
              <p className="text-foreground-secondary">
                Executions that failed in last 24h. View error messages to
                identify patterns. Spike in failures indicates system issues.
              </p>
            </div>
            <div>
              {/* oxlint-disable-next-line design-system/no-raw-colors -- pink has no role; red would change the hue (deferred) */}
              <p className="font-semibold text-pink-700">
                🩷 Invalid States (Data Corruption):
              </p>
              <p className="text-foreground-secondary">
                Executions in impossible states (QUEUED with startedAt, RUNNING
                without startedAt). Indicates DB corruption, race conditions, or
                crashes. Each requires manual investigation - no bulk actions
                provided.
              </p>
            </div>
            <div>
              <p className="font-semibold">Throughput Metrics:</p>
              <p className="text-foreground-secondary">
                Completions per hour shows system productivity. Declining
                throughput indicates performance degradation or executor issues.
              </p>
            </div>
            <div>
              <p className="font-semibold">Queue Health:</p>
              <p className="text-foreground-secondary">
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
