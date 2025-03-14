// app/admin/agents/data-table.tsx
"use client";

import { useState } from "react";
import { ColumnDef } from "@tanstack/react-table";
import { DataTable } from "@/components/ui/data-table";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  CheckCircle,
  XCircle,
  ExternalLink,
} from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
  DialogFooter,
} from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { StoreSubmission } from "@/lib/autogpt-server-api/types";
import { approveAgent, rejectAgent } from "@/app/admin/agents/actions";

export function AdminAgentsDataTable({ data }: { data: StoreSubmission[] }) {
  const [selectedAgent, setSelectedAgent] = useState<StoreSubmission | null>(null);
  const [isApproveDialogOpen, setIsApproveDialogOpen] = useState(false);
  const [isRejectDialogOpen, setIsRejectDialogOpen] = useState(false);

  const columns: ColumnDef<StoreSubmission>[] = [
    {
      accessorKey: "name",
      header: "Name",
      cell: ({ row }) => (
        <div className="font-medium">{row.getValue("name")}</div>
      ),
    },
    {
      accessorKey: "creator",
      header: "Creator",
      cell: ({ row }) => <div className="text-sm text-gray-500">
        {/* In a real implementation, you'd have creator email here */}
        {row.original.agent_id.split('-')[0] + '@example.com'}
      </div>,
    },
    {
      accessorKey: "description",
      header: "Description",
      cell: ({ row }) => (
        <div className="text-sm max-w-md truncate">{row.original.description}</div>
      ),
    },
    {
      accessorKey: "status",
      header: "Status",
      cell: ({ row }) => {
        const status = row.getValue("status") as string;
        return (
          <Badge
            className={
              status === "APPROVED" ? "bg-green-100 text-green-800" :
                status === "PENDING" ? "bg-yellow-100 text-yellow-800" :
                  "bg-red-100 text-red-800"
            }
          >
            {status.charAt(0) + status.slice(1).toLowerCase()}
          </Badge>
        );
      },
    },
    {
      accessorKey: "date_submitted",
      header: "Created",
      cell: ({ row }) => {
        const date = new Date(row.original.date_submitted);
        return (
          <div className="text-sm">
            {`${date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}, ${date.toLocaleTimeString('en-US', { hour: 'numeric', minute: 'numeric', hour12: true })}`}
          </div>
        );
      },
    },
    {
      id: "lastAction",
      header: "Last Action",
      cell: ({ row }) => {
        // This would ideally come from your API data
        const lastActionDate = row.original.reviewed_at ? new Date(row.original.reviewed_at) : null;

        if (!lastActionDate) {
          return (
            <div className="text-sm">
              <div>Awaiting review</div>
            </div>
          );
        }

        return (
          <div className="text-sm">
            <div>{`${lastActionDate.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })}, ${lastActionDate.toLocaleTimeString('en-US', { hour: 'numeric', minute: 'numeric', hour12: true })}`}</div>
            <div className="text-xs text-gray-500">nick@agpt.co</div>
            {row.original.status === "APPROVED" ? (
              <div className="text-xs text-gray-500">{row.original.review_comments}</div>
            ) : (
                <div className="text-xs text-gray-500">{row.original.review_comments}</div>
            )}
          </div>
        );
      },
    },
    {
      id: "actions",
      header: "Actions",
      cell: ({ row }) => {
        return (
          <div className="flex space-x-2">
            <Button variant="outline" size="sm" className="h-8 px-2 lg:px-3">
              <ExternalLink className="h-4 w-4 mr-2" />
              Builder
            </Button>

            {row.original.status === "PENDING" && (
              <>
                <Button
                  variant="outline"
                  size="sm"
                  className="h-8 px-2 lg:px-3 text-green-600 hover:text-green-700 hover:bg-green-50"
                  onClick={() => {
                    setSelectedAgent(row.original);
                    setIsApproveDialogOpen(true);
                  }}
                >
                  <CheckCircle className="h-4 w-4 mr-2" />
                  Approve
                </Button>

                <Button
                  variant="outline"
                  size="sm"
                  className="h-8 px-2 lg:px-3 text-red-600 hover:text-red-700 hover:bg-red-50"
                  onClick={() => {
                    setSelectedAgent(row.original);
                    setIsRejectDialogOpen(true);
                  }}
                >
                  <XCircle className="h-4 w-4 mr-2" />
                  Reject
                </Button>
              </>
            )}
          </div>
        );
      },
    },
  ];

  return (
    <div>
      <DataTable
        columns={columns}
        data={data}
        filterPlaceholder="Search agents by name, creator, or description..."
        filterColumn="name"
      />

      {/* Approve Dialog */}
      <Dialog open={isApproveDialogOpen} onOpenChange={setIsApproveDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Approve Agent</DialogTitle>
            <DialogDescription>
              Are you sure you want to approve this agent? This will make it available in the marketplace.
            </DialogDescription>
          </DialogHeader>

          <form action={approveAgent}>
            <input
              type="hidden"
              name="id"
              value={selectedAgent?.store_listing_version_id || ""}
            />

            <div className="grid gap-4 py-4">
              <div className="grid gap-2">
                <Label htmlFor="comments">Comments (Optional)</Label>
                <Textarea
                  id="comments"
                  name="comments"
                  placeholder="Add any comments for the agent creator"
                />
              </div>
            </div>

            <DialogFooter>
              <Button
                type="button"
                variant="outline"
                onClick={() => setIsApproveDialogOpen(false)}
              >
                Cancel
              </Button>
              <Button
                type="submit"
                onClick={() => setIsApproveDialogOpen(false)}
              >
                Approve
              </Button>
            </DialogFooter>
          </form>
        </DialogContent>
      </Dialog>

      {/* Reject Dialog */}
      <Dialog open={isRejectDialogOpen} onOpenChange={setIsRejectDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Reject Agent</DialogTitle>
            <DialogDescription>
              Please provide feedback on why this agent is being rejected.
            </DialogDescription>
          </DialogHeader>

          <form action={rejectAgent}>
            <input
              type="hidden"
              name="id"
              value={selectedAgent?.store_listing_version_id || ""}
            />

            <div className="grid gap-4 py-4">
              <div className="grid gap-2">
                <Label htmlFor="comments">Comments for Creator</Label>
                <Textarea
                  id="comments"
                  name="comments"
                  placeholder="Provide feedback for the agent creator"
                  required
                />
              </div>

              <div className="grid gap-2">
                <Label htmlFor="internal_comments">Internal Comments</Label>
                <Textarea
                  id="internal_comments"
                  name="internal_comments"
                  placeholder="Add any internal notes (not visible to creator)"
                />
              </div>
            </div>

            <DialogFooter>
              <Button
                type="button"
                variant="outline"
                onClick={() => setIsRejectDialogOpen(false)}
              >
                Cancel
              </Button>
              <Button
                type="submit"
                variant="destructive"
                onClick={() => setIsRejectDialogOpen(false)}
              >
                Reject
              </Button>
            </DialogFooter>
          </form>
        </DialogContent>
      </Dialog>
    </div>
  );
}