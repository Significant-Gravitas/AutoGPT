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
import { ArrowClockwise, Trash, Copy } from "@phosphor-icons/react";
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
  useGetV2ListAllUserSchedules,
  useGetV2ListOrphanedSchedules,
  usePostV2CleanupOrphanedSchedules,
} from "@/app/api/__generated__/endpoints/admin/admin";
import {
  TabsLine,
  TabsLineContent,
  TabsLineList,
  TabsLineTrigger,
} from "@/components/molecules/TabsLine/TabsLine";

interface ScheduleDetail {
  schedule_id: string;
  schedule_name: string;
  graph_id: string;
  graph_name: string;
  graph_version: number;
  user_id: string;
  user_email: string | null;
  cron: string;
  timezone: string;
  next_run_time: string;
}

interface OrphanedScheduleDetail {
  schedule_id: string;
  schedule_name: string;
  graph_id: string;
  graph_version: number;
  user_id: string;
  orphan_reason: string;
  error_detail: string | null;
  next_run_time: string;
}

interface SchedulesTableProps {
  onRefresh?: () => void;
  diagnosticsData?: {
    total_orphaned: number;
    user_schedules: number;
  };
}

export function SchedulesTable({
  onRefresh,
  diagnosticsData,
}: SchedulesTableProps) {
  const [activeTab, setActiveTab] = useState<"all" | "orphaned">("all");
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize] = useState(10);

  // Fetch data based on active tab
  const allSchedulesQuery = useGetV2ListAllUserSchedules(
    {
      limit: pageSize,
      offset: (currentPage - 1) * pageSize,
    },
    { query: { enabled: activeTab === "all" } },
  );

  const orphanedSchedulesQuery = useGetV2ListOrphanedSchedules({
    query: { enabled: activeTab === "orphaned" },
  });

  const activeQuery =
    activeTab === "orphaned" ? orphanedSchedulesQuery : allSchedulesQuery;

  const { data: schedulesResponse, isLoading, error, refetch } = activeQuery;

  const schedules =
    (schedulesResponse?.data as any)?.schedules || ([] as any[]);
  const total = (schedulesResponse?.data as any)?.total || 0;

  // Cleanup mutation
  const { mutateAsync: cleanupOrphanedSchedules, isPending: isDeleting } =
    usePostV2CleanupOrphanedSchedules();

  const handleSelectAll = (checked: boolean) => {
    if (checked) {
      setSelectedIds(new Set(schedules.map((s: any) => s.schedule_id)));
    } else {
      setSelectedIds(new Set());
    }
  };

  const handleSelectSchedule = (id: string, checked: boolean) => {
    const newSelected = new Set(selectedIds);
    if (checked) {
      newSelected.add(id);
    } else {
      newSelected.delete(id);
    }
    setSelectedIds(newSelected);
  };

  const confirmDelete = () => {
    setShowDeleteDialog(true);
  };

  const handleDelete = async () => {
    setShowDeleteDialog(false);

    try {
      const idsToDelete =
        activeTab === "orphaned" && selectedIds.size === 0
          ? schedules.map((s: any) => s.schedule_id)
          : Array.from(selectedIds);

      const result = await cleanupOrphanedSchedules({
        data: { execution_ids: idsToDelete }, // Reuses execution_ids field name
      });

      toast({
        title: "Success",
        description:
          (result.data as any)?.message ||
          `Deleted ${(result.data as any)?.deleted_count || 0} schedule(s)`,
      });

      setSelectedIds(new Set());
      await refetch();
      if (onRefresh) onRefresh();
    } catch (error: any) {
      console.error("Error deleting schedules:", error);
      toast({
        title: "Error",
        description: error.message || "Failed to delete schedules",
        variant: "destructive",
      });
    }
  };

  const totalPages = Math.ceil(total / pageSize);

  return (
    <>
      <Card>
        <TabsLine
          value={activeTab}
          onValueChange={(v) => setActiveTab(v as any)}
        >
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle>Schedules</CardTitle>
              <div className="flex gap-2">
                {activeTab === "orphaned" && schedules.length > 0 && (
                  <Button
                    variant="destructive"
                    size="small"
                    onClick={confirmDelete}
                    disabled={isDeleting}
                  >
                    <Trash className="mr-2 h-4 w-4" />
                    Delete All Orphaned ({total})
                  </Button>
                )}
                {selectedIds.size > 0 && (
                  <Button
                    variant="destructive"
                    size="small"
                    onClick={confirmDelete}
                    disabled={isDeleting}
                  >
                    <Trash className="mr-2 h-4 w-4" />
                    Delete Selected ({selectedIds.size})
                  </Button>
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

            <TabsLineList className="px-6">
              <TabsLineTrigger value="all">
                All Schedules
                {diagnosticsData && ` (${diagnosticsData.user_schedules})`}
              </TabsLineTrigger>
              <TabsLineTrigger value="orphaned">
                Orphaned
                {diagnosticsData && ` (${diagnosticsData.total_orphaned})`}
              </TabsLineTrigger>
            </TabsLineList>
          </CardHeader>

          <TabsLineContent value={activeTab}>
            <CardContent>
              {isLoading && schedules.length === 0 ? (
                <div className="flex h-32 items-center justify-center">
                  <ArrowClockwise className="h-6 w-6 animate-spin text-gray-400" />
                </div>
              ) : schedules.length === 0 ? (
                <div className="py-8 text-center text-gray-500">
                  No schedules found
                </div>
              ) : (
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead className="w-12">
                        <Checkbox
                          checked={
                            selectedIds.size === schedules.length &&
                            schedules.length > 0
                          }
                          onCheckedChange={handleSelectAll}
                        />
                      </TableHead>
                      <TableHead>Name</TableHead>
                      <TableHead>Graph</TableHead>
                      <TableHead>User</TableHead>
                      <TableHead>Cron</TableHead>
                      <TableHead>Next Run</TableHead>
                      {activeTab === "orphaned" && (
                        <TableHead>Orphan Reason</TableHead>
                      )}
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {schedules.map((schedule: any) => {
                      const isOrphaned = activeTab === "orphaned";
                      return (
                        <TableRow
                          key={schedule.schedule_id}
                          className={isOrphaned ? "bg-purple-50" : ""}
                        >
                          <TableCell>
                            <Checkbox
                              checked={selectedIds.has(schedule.schedule_id)}
                              onCheckedChange={(checked) =>
                                handleSelectSchedule(
                                  schedule.schedule_id,
                                  checked as boolean,
                                )
                              }
                            />
                          </TableCell>
                          <TableCell>{schedule.schedule_name}</TableCell>
                          <TableCell>
                            <div>{schedule.graph_name || "Unknown"}</div>
                            <div className="font-mono text-xs text-gray-500">
                              v{schedule.graph_version}
                            </div>
                          </TableCell>
                          <TableCell>
                            <div>
                              {(schedule as ScheduleDetail).user_email || (
                                <span className="text-gray-400">Unknown</span>
                              )}
                            </div>
                            <div
                              className="group flex cursor-pointer items-center gap-1 font-mono text-xs text-gray-500 hover:text-gray-700"
                              onClick={() => {
                                navigator.clipboard.writeText(schedule.user_id);
                                toast({
                                  title: "Copied",
                                  description: "User ID copied to clipboard",
                                });
                              }}
                              title="Click to copy user ID"
                            >
                              {schedule.user_id.substring(0, 8)}...
                              <Copy className="h-3 w-3 opacity-0 transition-opacity group-hover:opacity-100" />
                            </div>
                          </TableCell>
                          <TableCell>
                            <code className="rounded bg-gray-100 px-2 py-1 text-xs">
                              {schedule.cron}
                            </code>
                            <div className="text-xs text-gray-500">
                              {schedule.timezone}
                            </div>
                          </TableCell>
                          <TableCell>
                            {schedule.next_run_time
                              ? new Date(
                                  schedule.next_run_time,
                                ).toLocaleString()
                              : "Not scheduled"}
                          </TableCell>
                          {activeTab === "orphaned" && (
                            <TableCell>
                              <span className="text-xs text-purple-600">
                                {(
                                  schedule as OrphanedScheduleDetail
                                ).orphan_reason?.replace(/_/g, " ") ||
                                  "unknown"}
                              </span>
                            </TableCell>
                          )}
                        </TableRow>
                      );
                    })}
                  </TableBody>
                </Table>
              )}

              {totalPages > 1 && activeTab === "all" && (
                <div className="mt-4 flex items-center justify-between">
                  <div className="text-sm text-gray-600">
                    Showing {(currentPage - 1) * pageSize + 1} to{" "}
                    {Math.min(currentPage * pageSize, total)} of {total}{" "}
                    schedules
                  </div>
                  <div className="flex gap-2">
                    <Button
                      variant="outline"
                      size="small"
                      onClick={() => setCurrentPage(currentPage - 1)}
                      disabled={currentPage === 1}
                    >
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
                    </Button>
                  </div>
                </div>
              )}
            </CardContent>
          </TabsLineContent>
        </TabsLine>
      </Card>

      <Dialog open={showDeleteDialog} onOpenChange={setShowDeleteDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Confirm Delete Schedules</DialogTitle>
            <DialogDescription>
              {activeTab === "orphaned" && selectedIds.size === 0 ? (
                <>
                  Are you sure you want to delete ALL {total} orphaned
                  schedules?
                  <br />
                  <br />
                  These schedules reference deleted graphs or graphs the user no
                  longer has access to. Deleting them is safe.
                </>
              ) : (
                <>
                  Are you sure you want to delete {selectedIds.size} selected
                  schedule(s)?
                  <br />
                  <br />
                  This will permanently remove the schedules from the system.
                </>
              )}
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button
              variant="outline"
              onClick={() => setShowDeleteDialog(false)}
            >
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={handleDelete}
              className="bg-red-600 hover:bg-red-700"
            >
              Delete Schedules
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
