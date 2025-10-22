"use client";

import React, { useState, useEffect } from "react";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { toast } from "@/components/ui/use-toast";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import {
  Stop,
  StopCircle,
  ArrowClockwise,
  CaretLeft,
  CaretRight,
} from "@phosphor-icons/react";
import { apiUrl } from "@/lib/autogpt-server-api";

interface RunningExecutionDetail {
  execution_id: string;
  graph_id: string;
  graph_name: string;
  graph_version: number;
  user_id: string;
  user_email: string | null;
  status: string;
  started_at: string | null;
  queue_status: string | null;
}

interface ExecutionsTableProps {
  onRefresh?: () => void;
}

export function ExecutionsTable({ onRefresh }: ExecutionsTableProps) {
  const [executions, setExecutions] = useState<RunningExecutionDetail[]>([]);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [isLoading, setIsLoading] = useState(false);
  const [isStopping, setIsStopping] = useState(false);
  const [showStopDialog, setShowStopDialog] = useState(false);
  const [stopTarget, setStopTarget] = useState<"single" | "selected" | "all">(
    "single",
  );
  const [singleStopId, setSingleStopId] = useState<string | null>(null);
  const [total, setTotal] = useState(0);
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize] = useState(10);

  const fetchExecutions = async () => {
    setIsLoading(true);
    try {
      const offset = (currentPage - 1) * pageSize;
      const response = await fetch(
        `${apiUrl}/admin/diagnostics/executions/running?limit=${pageSize}&offset=${offset}`,
        {
          credentials: "include",
        },
      );

      if (!response.ok) {
        throw new Error("Failed to fetch executions");
      }

      const data = await response.json();
      setExecutions(data.executions || []);
      setTotal(data.total || 0);
    } catch (error) {
      console.error("Error fetching executions:", error);
      toast({
        title: "Error",
        description: "Failed to fetch running executions",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchExecutions();
  }, [currentPage]);

  const handleSelectAll = (checked: boolean) => {
    if (checked) {
      setSelectedIds(new Set(executions.map((e) => e.execution_id)));
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
    singleId?: string,
  ) => {
    setStopTarget(target);
    setSingleStopId(singleId || null);
    setShowStopDialog(true);
  };

  const handleStop = async () => {
    setShowStopDialog(false);
    setIsStopping(true);

    let endpoint: string;
    let body: any;

    if (stopTarget === "single" && singleStopId) {
      endpoint = `${apiUrl}/admin/diagnostics/executions/stop`;
      body = { execution_id: singleStopId };
    } else {
      let idsToStop: string[] = [];
      if (stopTarget === "selected") {
        idsToStop = Array.from(selectedIds);
      } else if (stopTarget === "all") {
        idsToStop = executions.map((e) => e.execution_id);
      }

      endpoint = `${apiUrl}/admin/diagnostics/executions/stop-bulk`;
      body = { execution_ids: idsToStop };
    }

    try {
      const response = await fetch(endpoint, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        credentials: "include",
        body: JSON.stringify(body),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Failed to stop executions");
      }

      toast({
        title: "Success",
        description: data.message || "Executions stopped successfully",
      });

      // Clear selections and refresh
      setSelectedIds(new Set());
      await fetchExecutions();
      if (onRefresh) {
        onRefresh();
      }
    } catch (error: any) {
      console.error("Error stopping executions:", error);
      toast({
        title: "Error",
        description: error.message || "Failed to stop executions",
        variant: "destructive",
      });
    } finally {
      setIsStopping(false);
    }
  };

  const totalPages = Math.ceil(total / pageSize);

  return (
    <>
      <Card>
        <CardHeader>
          <div className="flex items-center justify-between">
            <CardTitle>Running Executions</CardTitle>
            <div className="flex gap-2">
              {selectedIds.size > 0 && (
                <Button
                  variant="destructive"
                  size="sm"
                  onClick={() => confirmStop("selected")}
                  disabled={isStopping}
                >
                  <StopCircle className="mr-2 h-4 w-4" />
                  Stop Selected ({selectedIds.size})
                </Button>
              )}
              <Button
                variant="destructive"
                size="sm"
                onClick={() => confirmStop("all")}
                disabled={isStopping || executions.length === 0}
              >
                <StopCircle className="mr-2 h-4 w-4" />
                Stop All
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => {
                  fetchExecutions();
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
        </CardHeader>
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
                    <TableHead>Started At</TableHead>
                    <TableHead className="w-20">Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {executions.map((execution) => (
                    <TableRow key={execution.execution_id}>
                      <TableCell>
                        <Checkbox
                          checked={selectedIds.has(execution.execution_id)}
                          onCheckedChange={(checked) =>
                            handleSelectExecution(
                              execution.execution_id,
                              checked as boolean,
                            )
                          }
                        />
                      </TableCell>
                      <TableCell className="font-mono text-xs">
                        {execution.execution_id.substring(0, 8)}...
                      </TableCell>
                      <TableCell>{execution.graph_name}</TableCell>
                      <TableCell>{execution.graph_version}</TableCell>
                      <TableCell>
                        <div>
                          {execution.user_email || (
                            <span className="text-gray-400">Unknown</span>
                          )}
                        </div>
                        <div className="font-mono text-xs text-gray-500">
                          {execution.user_id.substring(0, 8)}...
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
                        {execution.started_at
                          ? new Date(execution.started_at).toLocaleString()
                          : "-"}
                      </TableCell>
                      <TableCell>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() =>
                            confirmStop("single", execution.execution_id)
                          }
                          disabled={isStopping}
                        >
                          <Stop className="h-4 w-4" />
                        </Button>
                      </TableCell>
                    </TableRow>
                  ))}
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
                      size="sm"
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
                      size="sm"
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
      </Card>

      <AlertDialog open={showStopDialog} onOpenChange={setShowStopDialog}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Confirm Stop Executions</AlertDialogTitle>
            <AlertDialogDescription>
              {stopTarget === "single" && (
                <>Are you sure you want to stop this execution?</>
              )}
              {stopTarget === "selected" && (
                <>
                  Are you sure you want to stop {selectedIds.size} selected
                  execution(s)?
                </>
              )}
              {stopTarget === "all" && (
                <>
                  Are you sure you want to stop ALL {executions.length} running
                  executions?
                </>
              )}
              <br />
              <br />
              This action cannot be undone. The executions will be marked as
              FAILED.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={handleStop}
              className="bg-red-600 hover:bg-red-700"
            >
              Stop Executions
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
}
