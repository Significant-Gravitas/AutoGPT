"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { CheckCircle, XCircle } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import type { StoreSubmission } from "@/lib/autogpt-server-api/types";
import { useRouter } from "next/navigation";
import {
  approveAgent,
  rejectAgent,
} from "@/app/(platform)/admin/marketplace/actions";
import { LoadingSpinner } from "@/components/ui/loading";

export function ApproveRejectButtons({
  version,
}: {
  version: StoreSubmission;
}) {
  const router = useRouter();
  const [isApproveDialogOpen, setIsApproveDialogOpen] = useState(false);
  const [isRejectDialogOpen, setIsRejectDialogOpen] = useState(false);
  const [isApproving, setIsApproving] = useState(false);
  const [isRejecting, setIsRejecting] = useState(false);

  const handleApproveSubmit = async (formData: FormData) => {
    setIsApproving(true);
    try {
      await approveAgent(formData);
      setIsApproveDialogOpen(false);
      router.refresh(); // Refresh the current route
    } catch (error) {
      console.error("Error approving agent:", error);
    } finally {
      setIsApproving(false);
    }
  };

  const handleRejectSubmit = async (formData: FormData) => {
    setIsRejecting(true);
    try {
      await rejectAgent(formData);
      setIsRejectDialogOpen(false);
      router.refresh(); // Refresh the current route
    } catch (error) {
      console.error("Error rejecting agent:", error);
    } finally {
      setIsRejecting(false);
    }
  };

  return (
    <>
      <div className="flex gap-2">
        <Button
          size="sm"
          onClick={() => setIsApproveDialogOpen(true)}
          className="bg-green-600 hover:bg-green-700"
          disabled={isApproving || isRejecting}
        >
          {isApproving ? (
            <>
              <LoadingSpinner className="mr-2 h-4 w-4" />
              Approving...
            </>
          ) : (
            <>
              <CheckCircle className="mr-2 h-4 w-4" />
              Approve
            </>
          )}
        </Button>
        <Button
          size="sm"
          variant="destructive"
          onClick={() => setIsRejectDialogOpen(true)}
          disabled={isApproving || isRejecting}
        >
          {isRejecting ? (
            <>
              <LoadingSpinner className="mr-2 h-4 w-4" />
              Rejecting...
            </>
          ) : (
            <>
              <XCircle className="mr-2 h-4 w-4" />
              Reject
            </>
          )}
        </Button>
      </div>

      {/* Approve Dialog */}
      <Dialog open={isApproveDialogOpen} onOpenChange={setIsApproveDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Approve Agent</DialogTitle>
            <DialogDescription>
              Are you sure you want to approve this agent? This will make it
              available in the marketplace.
            </DialogDescription>
          </DialogHeader>

          <form action={handleApproveSubmit}>
            <input
              type="hidden"
              name="id"
              value={version.store_listing_version_id || ""}
            />

            <div className="grid gap-4 py-4">
              <div className="grid gap-2">
                <Label htmlFor="comments">Comments (Optional)</Label>
                <Textarea
                  id="comments"
                  name="comments"
                  placeholder="Add any comments for the agent creator"
                  defaultValue="Meets all requirements"
                  disabled={isApproving}
                />
              </div>
            </div>

            <DialogFooter>
              <Button
                type="button"
                variant="outline"
                onClick={() => setIsApproveDialogOpen(false)}
                disabled={isApproving}
              >
                Cancel
              </Button>
              <Button type="submit" disabled={isApproving}>
                {isApproving ? (
                  <>
                    <LoadingSpinner className="mr-2 h-4 w-4" />
                    Approving...
                  </>
                ) : (
                  "Approve"
                )}
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

          <form action={handleRejectSubmit}>
            <input
              type="hidden"
              name="id"
              value={version.store_listing_version_id || ""}
            />

            <div className="grid gap-4 py-4">
              <div className="grid gap-2">
                <Label htmlFor="comments">Comments for Creator</Label>
                <Textarea
                  id="comments"
                  name="comments"
                  placeholder="Provide feedback for the agent creator"
                  required
                  disabled={isRejecting}
                />
              </div>

              <div className="grid gap-2">
                <Label htmlFor="internal_comments">Internal Comments</Label>
                <Textarea
                  id="internal_comments"
                  name="internal_comments"
                  placeholder="Add any internal notes (not visible to creator)"
                  disabled={isRejecting}
                />
              </div>
            </div>

            <DialogFooter>
              <Button
                type="button"
                variant="outline"
                onClick={() => setIsRejectDialogOpen(false)}
                disabled={isRejecting}
              >
                Cancel
              </Button>
              <Button
                type="submit"
                variant="destructive"
                disabled={isRejecting}
              >
                {isRejecting ? (
                  <>
                    <LoadingSpinner className="mr-2 h-4 w-4" />
                    Rejecting...
                  </>
                ) : (
                  "Reject"
                )}
              </Button>
            </DialogFooter>
          </form>
        </DialogContent>
      </Dialog>
    </>
  );
}
