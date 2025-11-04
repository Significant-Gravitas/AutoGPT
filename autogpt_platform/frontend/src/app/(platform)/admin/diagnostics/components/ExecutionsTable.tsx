"use client";

import { Button } from "@/components/atoms/Button/Button";
import { Card } from "@/components/atoms/Card/Card";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/__legacy__/ui/dialog";
import { toast } from "@/components/molecules/Toast/use-toast";
import {
  StopCircleIcon,
  ArrowClockwise,
  Stop,
  CaretLeft,
  CaretRight,
  Copy,
} from "@phosphor-icons/react";
import React, { useState } from "react";
import {
  Table,
  TableHeader,
  TableBody,
  TableHead,
  TableRow,
  TableCell,
} from "@/components/__legacy__/ui/table";
import { Checkbox } from "@/components/__legacy__/ui/checkbox";
import {
  CardHeader,
  CardTitle,
  CardContent,
} from "@/components/__legacy__/ui/card";
import {
  useGetV2ListRunningExecutions,
  useGetV2ListOrphanedExecutions,
  useGetV2ListFailedExecutions,
  useGetV2ListLongRunningExecutions,
  useGetV2ListStuckQueuedExecutions,
  usePostV2StopSingleExecution,
  usePostV2StopMultipleExecutions,
  usePostV2StopAllLongRunningExecutions,
  usePostV2CleanupOrphanedExecutions,
  usePostV2CleanupAllOrphanedExecutions,
  usePostV2CleanupAllStuckQueuedExecutions,
  usePostV2RequeueStuckExecution,
  usePostV2RequeueMultipleStuckExecutions,
  usePostV2RequeueAllStuckQueuedExecutions,
} from "@/app/api/__generated__/endpoints/admin/admin";
import {
  TabsLine,
  TabsLineContent,
  TabsLineList,
  TabsLineTrigger,
} from "@/components/molecules/TabsLine/TabsLine";

interface RunningExecutionDetail {
  execution_id: string;
  graph_id: string;
  graph_name: string;
  graph_version: number;
  user_id: string;
  user_email: string | null;
  status: string;
  created_at: string;
  started_at: string | null;
  queue_status: string | null;
}

interface ExecutionsTableProps {
  onRefresh?: () => void;
  initialTab?: "all" | "orphaned" | "failed" | "long-running" | "stuck-queued";
  onTabChange?: (
    tab: "all" | "orphaned" | "failed" | "long-running" | "stuck-queued",
  ) => void;
  diagnosticsData?: {
    orphaned_running: number;
    orphaned_queued: number;
    failed_count_24h: number;
    stuck_running_24h: number;
    stuck_queued_1h: number;
  };
}

export function ExecutionsTable({
  onRefresh,
  initialTab = "all",
  onTabChange,
  diagnosticsData,
}: ExecutionsTableProps) {
  const [activeTab, setActiveTab] = useState<
    "all" | "orphaned" | "failed" | "long-running" | "stuck-queued"
  >(initialTab);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [showStopDialog, setShowStopDialog] = useState(false);
  const [stopTarget, setStopTarget] = useState<"single" | "selected" | "all">(
    "single",
  );
  const [stopMode, setStopMode] = useState<"stop" | "cleanup" | "requeue">(
    "stop",
  );
  const [singleStopId, setSingleStopId] = useState<string | null>(null);
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize] = useState(10);

  // Handle tab changes
  function handleTabChange(newTab: string) {
    const tab = newTab as
      | "all"
      | "orphaned"
      | "failed"
      | "long-running"
      | "stuck-queued";
    setActiveTab(tab);
    setCurrentPage(1); // Reset to first page
    setSelectedIds(new Set()); // Clear selections
    if (onTabChange) onTabChange(tab);
  }

  // Sync with external tab changes (from clicking alert cards)
  React.useEffect(() => {
    if (initialTab !== activeTab) {
      setActiveTab(initialTab);
      setCurrentPage(1);
      setSelectedIds(new Set());
    }
  }, [initialTab]);

  // Fetch data based on active tab
  const runningQuery = useGetV2ListRunningExecutions(
    {
      limit: pageSize,
      offset: (currentPage - 1) * pageSize,
    },
    { query: { enabled: activeTab === "all" } },
  );

  const orphanedQuery = useGetV2ListOrphanedExecutions(
    {
      limit: pageSize,
      offset: (currentPage - 1) * pageSize,
    },
    { query: { enabled: activeTab === "orphaned" } },
  );

  const failedQuery = useGetV2ListFailedExecutions(
    {
      limit: pageSize,
      offset: (currentPage - 1) * pageSize,
      hours: 24,
    },
    { query: { enabled: activeTab === "failed" } },
  );

  // Long-running has dedicated endpoint (RUNNING status >24h only)
  const longRunningQuery = useGetV2ListLongRunningExecutions(
    {
      limit: pageSize,
      offset: (currentPage - 1) * pageSize,
    },
    { query: { enabled: activeTab === "long-running" } },
  );

  // Stuck queued has dedicated endpoint (QUEUED >1h)
  const stuckQueuedQuery = useGetV2ListStuckQueuedExecutions(
    {
      limit: pageSize,
      offset: (currentPage - 1) * pageSize,
    },
    { query: { enabled: activeTab === "stuck-queued" } },
  );

  // Select active query based on tab
  const activeQuery =
    activeTab === "orphaned"
      ? orphanedQuery
      : activeTab === "failed"
        ? failedQuery
        : activeTab === "long-running"
          ? longRunningQuery
          : activeTab === "stuck-queued"
            ? stuckQueuedQuery
            : runningQuery;

  const { data: executionsResponse, isLoading, error, refetch } = activeQuery;

  const executions =
    (executionsResponse?.data as any)?.executions ||
    ([] as RunningExecutionDetail[]);
  const total = (executionsResponse?.data as any)?.total || 0;

  // Stop single execution mutation
  const { mutateAsync: stopSingleExecution, isPending: isStoppingSingle } =
    usePostV2StopSingleExecution();

  // Stop multiple executions mutation
  const { mutateAsync: stopMultipleExecutions, isPending: isStoppingMultiple } =
    usePostV2StopMultipleExecutions();

  // Cleanup orphaned executions mutation
  const { mutateAsync: cleanupOrphanedExecutions, isPending: isCleaningUp } =
    usePostV2CleanupOrphanedExecutions();

  // Requeue stuck queued executions mutation
  const { mutateAsync: requeueSingleExecution, isPending: isRequeuingSingle } =
    usePostV2RequeueStuckExecution();

  const {
    mutateAsync: requeueMultipleExecutions,
    isPending: isRequeueingMultiple,
  } = usePostV2RequeueMultipleStuckExecutions();

  const { mutateAsync: requeueAllStuck, isPending: isRequeueingAll } =
    usePostV2RequeueAllStuckQueuedExecutions();

  const { mutateAsync: cleanupAllOrphaned, isPending: isCleaningUpAll } =
    usePostV2CleanupAllOrphanedExecutions();

  const {
    mutateAsync: cleanupAllStuckQueued,
    isPending: isCleaningUpAllStuckQueued,
  } = usePostV2CleanupAllStuckQueuedExecutions();

  const {
    mutateAsync: stopAllLongRunning,
    isPending: isStoppingAllLongRunning,
  } = usePostV2StopAllLongRunningExecutions();

  const isStopping =
    isStoppingSingle ||
    isStoppingMultiple ||
    isCleaningUp ||
    isRequeuingSingle ||
    isRequeueingMultiple ||
    isRequeueingAll ||
    isCleaningUpAll ||
    isCleaningUpAllStuckQueued ||
    isStoppingAllLongRunning;

  // Calculate which executions are orphaned (>24h old based on created_at)
  const now = new Date();
  const orphanedIds = new Set(
    executions
      .filter((e: RunningExecutionDetail) => {
        const createdDate = new Date(e.created_at);
        const ageHours =
          (now.getTime() - createdDate.getTime()) / (1000 * 60 * 60);
        return ageHours > 24;
      })
      .map((e: RunningExecutionDetail) => e.execution_id),
  );

  const selectedOrphanedIds = Array.from(selectedIds).filter((id) =>
    orphanedIds.has(id),
  );
  const hasOrphanedSelected = selectedOrphanedIds.length > 0;

  // Show error toast if fetching fails (in useEffect to avoid render side-effects)
  React.useEffect(() => {
    if (error) {
      toast({
        title: "Error",
        description: "Failed to fetch executions",
        variant: "destructive",
      });
    }
  }, [error]);

  const handleSelectAll = (checked: boolean) => {
    if (checked) {
      setSelectedIds(
        new Set(executions.map((e: RunningExecutionDetail) => e.execution_id)),
      );
    } else {
      setSelectedIds(new Set());
    }
  };

  const handleSelectExecution = (id: string, checked: boolean) => {
    const newSelected = new Set(selectedIds);
    if (checked) {
      newSelected.add(id);
    } else {
      newSelected.delete(id);
    }
    setSelectedIds(newSelected);
  };

  const confirmStop = (
    target: "single" | "selected" | "all",
    mode: "stop" | "cleanup" | "requeue",
    singleId?: string,
  ) => {
    setStopTarget(target);
    setStopMode(mode);
    setSingleStopId(singleId || null);
    setShowStopDialog(true);
  };

  const handleStop = async () => {
    setShowStopDialog(false);

    try {
      if (stopTarget === "single" && singleStopId) {
        // Single execution - use appropriate method
        const result =
          stopMode === "cleanup"
            ? await cleanupOrphanedExecutions({
                data: { execution_ids: [singleStopId] },
              })
            : stopMode === "requeue"
              ? await requeueSingleExecution({
                  data: { execution_id: singleStopId },
                })
              : await stopSingleExecution({
                  data: { execution_id: singleStopId },
                });

        toast({
          title: "Success",
          description:
            (result.data as any)?.message ||
            (stopMode === "cleanup"
              ? "Orphaned execution cleaned up"
              : stopMode === "requeue"
                ? "Execution requeued"
                : "Execution stopped"),
        });
      } else {
        // Multiple executions
        if (stopMode === "requeue") {
          // Requeue stuck queued executions
          if (stopTarget === "all") {
            // Use ALL endpoint for entire dataset
            const result = await requeueAllStuck();

            toast({
              title: "Success",
              description:
                (result.data as any)?.message ||
                `Requeued ${(result.data as any)?.requeued_count || 0} stuck executions`,
            });
          } else {
            // Selected only
            const allIds = Array.from(selectedIds);
            const result = await requeueMultipleExecutions({
              data: { execution_ids: allIds },
            });

            toast({
              title: "Success",
              description:
                (result.data as any)?.message ||
                `Requeued ${(result.data as any)?.requeued_count || 0} execution(s)`,
            });
          }
        } else if (stopMode === "cleanup") {
          // Cleanup executions
          if (stopTarget === "all" && activeTab === "orphaned") {
            // Use ALL endpoint for orphaned tab (>24h old)
            const result = await cleanupAllOrphaned();

            toast({
              title: "Success",
              description:
                (result.data as any)?.message ||
                `Cleaned up ${(result.data as any)?.stopped_count || 0} orphaned executions`,
            });
          } else if (stopTarget === "all" && activeTab === "stuck-queued") {
            // Use ALL endpoint for stuck-queued tab (>1h old)
            const result = await cleanupAllStuckQueued();

            toast({
              title: "Success",
              description:
                (result.data as any)?.message ||
                `Cleaned up ${(result.data as any)?.stopped_count || 0} stuck queued executions`,
            });
          } else {
            // Selected or other tabs
            const allIds =
              stopTarget === "selected"
                ? Array.from(selectedIds)
                : executions.map((e: RunningExecutionDetail) => e.execution_id);

            const result = await cleanupOrphanedExecutions({
              data: { execution_ids: allIds },
            });

            toast({
              title: "Success",
              description:
                (result.data as any)?.message ||
                `Cleaned up ${(result.data as any)?.stopped_count || 0} execution(s)`,
            });
          }
        } else {
          // Stop - handle long-running ALL or split active/orphaned
          if (stopTarget === "all" && activeTab === "long-running") {
            // Use ALL endpoint for long-running tab
            const result = await stopAllLongRunning();

            toast({
              title: "Success",
              description:
                (result.data as any)?.message ||
                `Stopped ${(result.data as any)?.stopped_count || 0} long-running executions`,
            });
          } else {
            // Stop selected - intelligently split between active and orphaned
            const activeIds: string[] = [];
            const orphanedIdsToCleanup: string[] = [];

            const allIds = Array.from(selectedIds);

            // Split into active vs orphaned
            allIds.forEach((id: string) => {
              if (orphanedIds.has(id)) {
                orphanedIdsToCleanup.push(id);
              } else {
                activeIds.push(id);
              }
            });

            // Execute both operations in parallel
            const results = await Promise.all([
              activeIds.length > 0
                ? stopMultipleExecutions({
                    data: { execution_ids: activeIds },
                  })
                : Promise.resolve(null),
              orphanedIdsToCleanup.length > 0
                ? cleanupOrphanedExecutions({
                    data: { execution_ids: orphanedIdsToCleanup },
                  })
                : Promise.resolve(null),
            ]);

            const stoppedCount = results[0]
              ? (results[0].data as any)?.stopped_count || 0
              : 0;
            const cleanedCount = results[1]
              ? (results[1].data as any)?.stopped_count || 0
              : 0;

            toast({
              title: "Success",
              description:
                stoppedCount > 0 && cleanedCount > 0
                  ? `Stopped ${stoppedCount} active and cleaned ${cleanedCount} orphaned executions`
                  : stoppedCount > 0
                    ? `Stopped ${stoppedCount} execution(s)`
                    : `Cleaned ${cleanedCount} orphaned execution(s)`,
            });
          }
        }
      }

      // Clear selections and refresh
      setSelectedIds(new Set());
      await refetch();
      if (onRefresh) {
        onRefresh();
      }
    } catch (error: any) {
      console.error("Error stopping/cleaning executions:", error);
      toast({
        title: "Error",
        description: error.message || "Failed to stop/cleanup executions",
        variant: "destructive",
      });
    }
  };

  const totalPages = Math.ceil(total / pageSize);

  return (
    <>
      <Card>
        <TabsLine value={activeTab} onValueChange={handleTabChange}>
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle>Executions</CardTitle>
              <div className="flex gap-2">
                {/* Show Cleanup and Requeue buttons for stuck-queued tab */}
                {activeTab === "stuck-queued" && total > 0 && (
                  <>
                    <Button
                      variant="outline"
                      size="small"
                      onClick={() => confirmStop("all", "cleanup")}
                      disabled={isStopping}
                      className="border-orange-500 text-orange-700 hover:bg-orange-50"
                    >
                      <StopCircleIcon className="mr-2 h-4 w-4" />
                      Cleanup All ({total})
                    </Button>
                    <Button
                      variant="outline"
                      size="small"
                      onClick={() => confirmStop("all", "requeue")}
                      disabled={isStopping}
                      className="border-blue-500 text-blue-700 hover:bg-blue-50"
                    >
                      <ArrowClockwise className="mr-2 h-4 w-4" />
                      Requeue All ({total})
                    </Button>
                  </>
                )}
                {selectedIds.size > 0 && activeTab !== "stuck-queued" && (
                  <Button
                    variant="destructive"
                    size="small"
                    onClick={() => confirmStop("selected", "stop")}
                    disabled={isStopping}
                  >
                    <StopCircleIcon className="mr-2 h-4 w-4" />
                    Stop Selected ({selectedIds.size})
                    {hasOrphanedSelected && (
                      <span className="ml-1 text-xs text-orange-200">
                        ({selectedOrphanedIds.length} orphaned)
                      </span>
                    )}
                  </Button>
                )}
                {/* Only show Stop All for specific tabs, not "all" tab */}
                {activeTab === "long-running" && total > 0 && (
                  <Button
                    variant="destructive"
                    size="small"
                    onClick={() => confirmStop("all", "stop")}
                    disabled={isStopping}
                  >
                    <StopCircleIcon className="mr-2 h-4 w-4" />
                    Stop All Long-Running ({total})
                  </Button>
                )}
                {activeTab === "failed" && selectedIds.size === 0 && (
                  <div className="px-3 text-sm text-gray-500">
                    View-only (select to delete)
                  </div>
                )}
                <Button
                  variant="outline"
                  size="small"
                  onClick={() => {
                    refetch();
                    if (onRefresh) onRefresh();
                  }}
                  disabled={isLoading}
                >
                  <ArrowClockwise
                    className={`h-4 w-4 ${isLoading ? "animate-spin" : ""}`}
                  />
                </Button>
              </div>
            </div>

            {/* Tabs for filtering */}
            <TabsLineList className="px-6">
              <TabsLineTrigger value="all">
                All
                {activeTab === "all" && ` (${total})`}
              </TabsLineTrigger>
              <TabsLineTrigger value="orphaned">
                Orphaned
                {diagnosticsData &&
                  ` (${diagnosticsData.orphaned_running + diagnosticsData.orphaned_queued})`}
              </TabsLineTrigger>
              <TabsLineTrigger value="stuck-queued">
                Stuck Queued
                {diagnosticsData && ` (${diagnosticsData.stuck_queued_1h})`}
              </TabsLineTrigger>
              <TabsLineTrigger value="long-running">
                Long-Running
                {diagnosticsData && ` (${diagnosticsData.stuck_running_24h})`}
              </TabsLineTrigger>
              <TabsLineTrigger value="failed">
                Failed
                {diagnosticsData && ` (${diagnosticsData.failed_count_24h})`}
              </TabsLineTrigger>
            </TabsLineList>
          </CardHeader>

          <TabsLineContent value={activeTab}>
            <CardContent>
              {isLoading && executions.length === 0 ? (
                <div className="flex h-32 items-center justify-center">
                  <ArrowClockwise className="h-6 w-6 animate-spin text-gray-400" />
                </div>
              ) : executions.length === 0 ? (
                <div className="py-8 text-center text-gray-500">
                  No running executions
                </div>
              ) : (
                <>
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead className="w-12">
                          <Checkbox
                            checked={
                              selectedIds.size === executions.length &&
                              executions.length > 0
                            }
                            onCheckedChange={handleSelectAll}
                          />
                        </TableHead>
                        <TableHead>Execution ID</TableHead>
                        <TableHead>Agent Name</TableHead>
                        <TableHead>Version</TableHead>
                        <TableHead>User</TableHead>
                        <TableHead>Status</TableHead>
                        <TableHead>Age</TableHead>
                        <TableHead>
                          {activeTab === "failed" ? "Failed At" : "Started At"}
                        </TableHead>
                        {activeTab === "failed" && (
                          <TableHead>Error Message</TableHead>
                        )}
                        <TableHead className="w-20">Actions</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {executions.map((execution: RunningExecutionDetail) => {
                        const isOrphaned = orphanedIds.has(
                          execution.execution_id,
                        );
                        return (
                          <TableRow
                            key={execution.execution_id}
                            className={
                              isOrphaned
                                ? "bg-orange-50 hover:bg-orange-100"
                                : ""
                            }
                          >
                            <TableCell>
                              <Checkbox
                                checked={selectedIds.has(
                                  execution.execution_id,
                                )}
                                onCheckedChange={(checked) =>
                                  handleSelectExecution(
                                    execution.execution_id,
                                    checked as boolean,
                                  )
                                }
                              />
                            </TableCell>
                            <TableCell className="font-mono text-xs">
                              <div
                                className="group flex cursor-pointer items-center gap-1 hover:text-gray-700"
                                onClick={() => {
                                  navigator.clipboard.writeText(
                                    execution.execution_id,
                                  );
                                  toast({
                                    title: "Copied",
                                    description:
                                      "Execution ID copied to clipboard",
                                  });
                                }}
                                title="Click to copy full execution ID"
                              >
                                {execution.execution_id.substring(0, 8)}...
                                <Copy className="h-3 w-3 opacity-0 transition-opacity group-hover:opacity-100" />
                              </div>
                            </TableCell>
                            <TableCell>{execution.graph_name}</TableCell>
                            <TableCell>{execution.graph_version}</TableCell>
                            <TableCell>
                              <div>
                                {execution.user_email || (
                                  <span className="text-gray-400">Unknown</span>
                                )}
                              </div>
                              <div
                                className="group flex cursor-pointer items-center gap-1 font-mono text-xs text-gray-500 hover:text-gray-700"
                                onClick={() => {
                                  navigator.clipboard.writeText(
                                    execution.user_id,
                                  );
                                  toast({
                                    title: "Copied",
                                    description: "User ID copied to clipboard",
                                  });
                                }}
                                title="Click to copy full user ID"
                              >
                                {execution.user_id.substring(0, 8)}...
                                <Copy className="h-3 w-3 opacity-0 transition-opacity group-hover:opacity-100" />
                              </div>
                            </TableCell>
                            <TableCell>
                              <span
                                className={`inline-flex rounded-full px-2 py-1 text-xs font-semibold ${
                                  execution.status === "RUNNING"
                                    ? "bg-green-100 text-green-800"
                                    : "bg-yellow-100 text-yellow-800"
                                }`}
                              >
                                {execution.status}
                              </span>
                            </TableCell>
                            <TableCell>
                              {(() => {
                                if (!execution.started_at)
                                  return "Never started";
                                const ageMs =
                                  now.getTime() -
                                  new Date(execution.started_at).getTime();
                                const ageHours = ageMs / (1000 * 60 * 60);
                                const ageDays = Math.floor(ageHours / 24);
                                const remainingHours = Math.floor(
                                  ageHours % 24,
                                );

                                if (ageDays > 0) {
                                  return (
                                    <span
                                      className={
                                        ageDays > 1
                                          ? "font-semibold text-orange-600"
                                          : ""
                                      }
                                    >
                                      {ageDays}d {remainingHours}h
                                    </span>
                                  );
                                } else {
                                  return `${remainingHours}h`;
                                }
                              })()}
                            </TableCell>
                            <TableCell>
                              {activeTab === "failed"
                                ? (execution as any).failed_at
                                  ? new Date(
                                      (execution as any).failed_at,
                                    ).toLocaleString()
                                  : "-"
                                : execution.started_at
                                  ? new Date(
                                      execution.started_at,
                                    ).toLocaleString()
                                  : "-"}
                            </TableCell>
                            {activeTab === "failed" && (
                              <TableCell className="max-w-xs truncate">
                                <span
                                  className="text-xs text-red-600"
                                  title={(execution as any).error_message || ""}
                                >
                                  {(execution as any).error_message ||
                                    "No error message"}
                                </span>
                              </TableCell>
                            )}
                            <TableCell>
                              <div className="flex gap-1">
                                {activeTab === "stuck-queued" ? (
                                  <>
                                    <Button
                                      variant="ghost"
                                      size="small"
                                      onClick={() =>
                                        confirmStop(
                                          "single",
                                          "cleanup",
                                          execution.execution_id,
                                        )
                                      }
                                      disabled={isStopping}
                                      className="text-orange-600 hover:bg-orange-50"
                                      title="Cleanup (mark as FAILED)"
                                    >
                                      <StopCircleIcon className="h-4 w-4" />
                                    </Button>
                                    <Button
                                      variant="ghost"
                                      size="small"
                                      onClick={() =>
                                        confirmStop(
                                          "single",
                                          "requeue",
                                          execution.execution_id,
                                        )
                                      }
                                      disabled={isStopping}
                                      className="text-blue-600 hover:bg-blue-50"
                                      title="Requeue (send to RabbitMQ)"
                                    >
                                      <ArrowClockwise className="h-4 w-4" />
                                    </Button>
                                  </>
                                ) : (
                                  <Button
                                    variant="ghost"
                                    size="small"
                                    onClick={() => {
                                      const isOrphaned = orphanedIds.has(
                                        execution.execution_id,
                                      );
                                      confirmStop(
                                        "single",
                                        isOrphaned ? "cleanup" : "stop",
                                        execution.execution_id,
                                      );
                                    }}
                                    disabled={isStopping}
                                    className={
                                      orphanedIds.has(execution.execution_id)
                                        ? "text-orange-600 hover:bg-orange-50"
                                        : ""
                                    }
                                  >
                                    <Stop className="h-4 w-4" />
                                  </Button>
                                )}
                              </div>
                            </TableCell>
                          </TableRow>
                        );
                      })}
                    </TableBody>
                  </Table>

                  {totalPages > 1 && (
                    <div className="mt-4 flex items-center justify-between">
                      <div className="text-sm text-gray-600">
                        Showing {(currentPage - 1) * pageSize + 1} to{" "}
                        {Math.min(currentPage * pageSize, total)} of {total}{" "}
                        executions
                      </div>
                      <div className="flex gap-2">
                        <Button
                          variant="outline"
                          size="small"
                          onClick={() => setCurrentPage(currentPage - 1)}
                          disabled={currentPage === 1}
                        >
                          <CaretLeft className="h-4 w-4" />
                          Previous
                        </Button>
                        <div className="flex items-center px-3">
                          Page {currentPage} of {totalPages}
                        </div>
                        <Button
                          variant="outline"
                          size="small"
                          onClick={() => setCurrentPage(currentPage + 1)}
                          disabled={currentPage === totalPages}
                        >
                          Next
                          <CaretRight className="h-4 w-4" />
                        </Button>
                      </div>
                    </div>
                  )}
                </>
              )}
            </CardContent>
          </TabsLineContent>
        </TabsLine>
      </Card>

      <Dialog open={showStopDialog} onOpenChange={setShowStopDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>
              {stopMode === "cleanup"
                ? "Confirm Cleanup Orphaned Executions"
                : stopMode === "requeue"
                  ? "Confirm Requeue Stuck Executions"
                  : "Confirm Stop Executions"}
            </DialogTitle>
            <DialogDescription>
              {stopMode === "requeue" ? (
                <>
                  {stopTarget === "all" && (
                    <>
                      Are you sure you want to requeue ALL {total} stuck
                      executions?
                    </>
                  )}
                  <br />
                  <br />
                  <strong className="text-blue-700">⚠️ Warning:</strong> This
                  will publish these executions to RabbitMQ to be processed
                  again. This <strong>will cost credits</strong> and may fail
                  again if the original issue persists.
                  <br />
                  <br />
                  Only requeue if you believe the executions are stuck due to a
                  temporary issue (executor restart, RabbitMQ purge, etc).
                </>
              ) : stopMode === "cleanup" ? (
                <>
                  {stopTarget === "single" && (
                    <>
                      Are you sure you want to cleanup this orphaned execution?
                    </>
                  )}
                  {stopTarget === "selected" && (
                    <>
                      Are you sure you want to cleanup{" "}
                      {selectedOrphanedIds.length} orphaned execution(s)?
                    </>
                  )}
                  {stopTarget === "all" && (
                    <>
                      Are you sure you want to cleanup ALL {orphanedIds.size}{" "}
                      orphaned executions?
                    </>
                  )}
                  <br />
                  <br />
                  <strong>Orphaned executions</strong> are {">"}24h old and not
                  actually running in the executor. This will mark them as
                  FAILED in the database only (no cancel signal sent).
                </>
              ) : (
                <>
                  {stopTarget === "single" && (
                    <>Are you sure you want to stop this execution?</>
                  )}
                  {stopTarget === "selected" && (
                    <>
                      Are you sure you want to stop {selectedIds.size} selected
                      execution(s)?
                      {hasOrphanedSelected && (
                        <>
                          <br />
                          <br />
                          <span className="text-orange-600">
                            Includes {selectedOrphanedIds.length} orphaned
                            execution(s) that will be cleaned up directly.
                          </span>
                        </>
                      )}
                    </>
                  )}
                  {stopTarget === "all" && (
                    <>
                      Are you sure you want to stop ALL {executions.length}{" "}
                      execution(s)?
                      {orphanedIds.size > 0 && (
                        <>
                          <br />
                          <br />
                          <span className="text-orange-600">
                            Includes {orphanedIds.size} orphaned execution(s) (
                            {">"}24h old) that will be cleaned up directly.
                          </span>
                        </>
                      )}
                    </>
                  )}
                  <br />
                  <br />
                  This will automatically:
                  <ul className="mt-2 list-disc pl-5 text-sm">
                    <li>Send cancel signals for active executions</li>
                    <li>
                      Clean up orphaned executions ({">"}24h old) directly in DB
                    </li>
                    <li>Mark all as FAILED</li>
                  </ul>
                </>
              )}
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button variant="outline" onClick={() => setShowStopDialog(false)}>
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={handleStop}
              className={
                stopMode === "cleanup"
                  ? "bg-orange-600 hover:bg-orange-700"
                  : stopMode === "requeue"
                    ? "bg-blue-600 hover:bg-blue-700"
                    : "bg-red-600 hover:bg-red-700"
              }
            >
              {stopMode === "cleanup"
                ? "Cleanup Orphaned"
                : stopMode === "requeue"
                  ? "Requeue Executions"
                  : "Stop Executions"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
