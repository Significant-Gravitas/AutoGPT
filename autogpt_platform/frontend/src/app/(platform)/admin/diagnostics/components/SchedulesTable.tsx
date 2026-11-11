"use client";

import { Button } from "@/components/atoms/Button/Button";
import {
  Card,
  CardHeader,
  CardTitle,
  CardContent,
} from "@/components/atoms/Card/Card";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/molecules/Dialog/components/DialogPrimitives";
import { toast } from "@/components/molecules/Toast/use-toast";
import React, { useState } from "react";
import {
  Table,
  TableHeader,
  TableBody,
  TableHead,
  TableRow,
  TableCell,
} from "@/components/molecules/Table/components/TablePrimitives";
import { Checkbox } from "@/components/atoms/Checkbox/Checkbox";
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
import {
  Copy01Icon,
  Delete02Icon,
  Refresh01Icon,
} from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";

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
  graph_name?: string;
  graph_version: number;
  user_id: string;
  user_email?: string | null;
  cron?: string;
  timezone?: string;
  orphan_reason: string;
  error_detail: string | null;
  next_run_time: string;
}

interface CleanupResponseData {
  success: boolean;
  message: string;
  deleted_count?: number;
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

  const {
    data: schedulesResponse,
    isLoading,
    error: _error,
    refetch,
  } = activeQuery;

  const schedulesData = schedulesResponse?.data as
    | { schedules: (ScheduleDetail | OrphanedScheduleDetail)[]; total: number }
    | undefined;
  const schedules = schedulesData?.schedules || [];
  const total = schedulesData?.total || 0;

  // Cleanup mutation
  const { mutateAsync: cleanupOrphanedSchedules, isPending: isDeleting } =
    usePostV2CleanupOrphanedSchedules();

  const handleSelectAll = (checked: boolean) => {
    if (checked) {
      setSelectedIds(
        new Set(
          schedules.map(
            (s: ScheduleDetail | OrphanedScheduleDetail) => s.schedule_id,
          ),
        ),
      );
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
          ? schedules.map(
              (s: ScheduleDetail | OrphanedScheduleDetail) => s.schedule_id,
            )
          : Array.from(selectedIds);

      const result = await cleanupOrphanedSchedules({
        data: { schedule_ids: idsToDelete },
      });

      toast({
        title: "Success",
        description:
          (result.data as CleanupResponseData)?.message ||
          `Deleted ${(result.data as CleanupResponseData)?.deleted_count || 0} schedule(s)`,
      });

      setSelectedIds(new Set());
      await refetch();
      if (onRefresh) onRefresh();
    } catch (err: unknown) {
      console.error("Error deleting schedules:", err);
      toast({
        title: "Error",
        description:
          err instanceof Error ? err.message : "Failed to delete schedules",
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
          onValueChange={(v) => setActiveTab(v as "all" | "orphaned")}
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
                    <Icon icon={Delete02Icon} className="mr-2 h-4 w-4" />
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
                    <Icon icon={Delete02Icon} className="mr-2 h-4 w-4" />
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
                  <span
                    className={cn("block h-4 w-4", isLoading && "animate-spin")}
                  >
                    <Icon icon={Refresh01Icon} className="h-4 w-4" />
                  </span>
                </Button>
              </div>
            </div>

            {/* oxlint-disable-next-line design-system/no-restyle -- TabsLineList has no inset variant; a padded wrapper would shorten its underline (src/components is P9's) */}
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
                  <span className="block h-6 w-6 animate-spin text-foreground-subtle">
                    <Icon icon={Refresh01Icon} className="h-6 w-6" />
                  </span>
                </div>
              ) : schedules.length === 0 ? (
                <div className="py-8 text-center text-foreground-tertiary">
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
                    {schedules.map(
                      (schedule: ScheduleDetail | OrphanedScheduleDetail) => {
                        const isOrphaned = activeTab === "orphaned";
                        return (
                          <TableRow
                            key={schedule.schedule_id}
                            // oxlint-disable-next-line design-system/no-restyle -- TableRow has no highlight variant (src/components is P9's)
                            className={isOrphaned ? "bg-accent-surface" : ""}
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
                              <Text
                                variant="small-mono-tight"
                                as="div"
                                tone="muted"
                              >
                                v{schedule.graph_version}
                              </Text>
                            </TableCell>
                            <TableCell>
                              <div>
                                {(schedule as ScheduleDetail).user_email || (
                                  <span className="text-foreground-subtle">
                                    Unknown
                                  </span>
                                )}
                              </div>
                              <div
                                className="group flex cursor-pointer items-center gap-1 text-foreground-tertiary hover:text-foreground-body"
                                onClick={() => {
                                  navigator.clipboard.writeText(
                                    schedule.user_id,
                                  );
                                  toast({
                                    title: "Copied",
                                    description: "User ID copied to clipboard",
                                  });
                                }}
                                title="Click to copy user ID"
                              >
                                <Text
                                  variant="small-mono-tight"
                                  as="span"
                                  tone="inherit"
                                >
                                  {schedule.user_id.substring(0, 8)}...
                                </Text>
                                <span className="block opacity-0 transition-opacity group-hover:opacity-100">
                                  <Icon icon={Copy01Icon} className="h-3 w-3" />
                                </span>
                              </div>
                            </TableCell>
                            <TableCell>
                              {schedule.cron ? (
                                <>
                                  <code className="rounded bg-surface-hover px-2 py-1 font-mono text-caption leading-4">
                                    {schedule.cron}
                                  </code>
                                  <Text
                                    variant="small-tight"
                                    as="div"
                                    tone="muted"
                                  >
                                    {schedule.timezone}
                                  </Text>
                                </>
                              ) : (
                                <span className="text-foreground-subtle">
                                  N/A
                                </span>
                              )}
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
                                <span className="text-caption leading-4 text-brand-hover">
                                  {(
                                    schedule as OrphanedScheduleDetail
                                  ).orphan_reason?.replace(/_/g, " ") ||
                                    "unknown"}
                                </span>
                              </TableCell>
                            )}
                          </TableRow>
                        );
                      },
                    )}
                  </TableBody>
                </Table>
              )}

              {totalPages > 1 && activeTab === "all" && (
                <div className="mt-4 flex items-center justify-between">
                  <Text variant="body-tight" as="div" tone="secondary">
                    Showing {(currentPage - 1) * pageSize + 1} to{" "}
                    {Math.min(currentPage * pageSize, total)} of {total}{" "}
                    schedules
                  </Text>
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
            <Button variant="destructive" onClick={handleDelete}>
              Delete Schedules
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  );
}
