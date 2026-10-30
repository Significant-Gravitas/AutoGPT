"use client";

import { useState } from "react";
import { Button } from "@/components/atoms/Button/Button";
import {
  CancelCircleIcon,
  CheckmarkCircle02Icon,
} from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/molecules/Dialog/components/DialogPrimitives";
import { Label } from "@/components/atoms/Label/Label";
import { Textarea } from "@/components/atoms/Textarea/Textarea";
import type { StoreSubmissionAdminView } from "@/app/api/__generated__/models/storeSubmissionAdminView";
import { useRouter } from "next/navigation";
import {
  approveAgent,
  rejectAgent,
} from "@/app/(platform)/admin/marketplace/actions";

export function ApproveRejectButtons({
  version,
}: {
  version: StoreSubmissionAdminView;
}) {
  const router = useRouter();
  const [isApproveDialogOpen, setIsApproveDialogOpen] = useState(false);
  const [isRejectDialogOpen, setIsRejectDialogOpen] = useState(false);

  const isApproved = version.status === "APPROVED";

  const handleApproveSubmit = async (formData: FormData) => {
    setIsApproveDialogOpen(false);
    try {
      await approveAgent(formData);
      router.refresh(); // Refresh the current route
    } catch (error) {
      console.error("Error approving agent:", error);
    }
  };

  const handleRejectSubmit = async (formData: FormData) => {
    setIsRejectDialogOpen(false);
    try {
      await rejectAgent(formData);
      router.refresh(); // Refresh the current route
    } catch (error) {
      console.error("Error rejecting agent:", error);
    }
  };

  return (
    <>
      {!isApproved && (
        <Button
          size="xs"
          variant="secondary"
          onClick={(e) => {
            e.stopPropagation();
            setIsApproveDialogOpen(true);
          }}
        >
          <span className="flex items-center gap-1.5 text-success">
            <Icon icon={CheckmarkCircle02Icon} className="h-4 w-4" />
            Approve
          </span>
        </Button>
      )}
      <Button
        size="xs"
        variant="secondary"
        onClick={(e) => {
          e.stopPropagation();
          setIsRejectDialogOpen(true);
        }}
      >
        <span className="flex items-center gap-1.5 text-error-hover">
          <Icon icon={CancelCircleIcon} className="h-4 w-4" />
          {isApproved ? "Revoke" : "Reject"}
        </span>
      </Button>

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
              value={version.listing_version_id || ""}
            />

            <div className="grid gap-4 py-4">
              <div className="grid gap-2">
                <Label htmlFor="comments">Comments (Optional)</Label>
                <Textarea
                  id="comments"
                  name="comments"
                  placeholder="Add any comments for the agent creator"
                  defaultValue="Meets all requirements"
                />
              </div>
            </div>

            <DialogFooter>
              <Button
                type="button"
                variant="compact-outline"
                size="compact"
                onClick={() => setIsApproveDialogOpen(false)}
              >
                Cancel
              </Button>
              <Button variant="compact-primary" size="compact" type="submit">
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
            <DialogTitle>
              {isApproved ? "Revoke Approved Agent" : "Reject Agent"}
            </DialogTitle>
            <DialogDescription>
              {isApproved
                ? "Are you sure you want to revoke approval for this agent? This will remove it from the marketplace."
                : "Please provide feedback on why this agent is being rejected."}
            </DialogDescription>
          </DialogHeader>

          <form action={handleRejectSubmit}>
            <input
              type="hidden"
              name="id"
              value={version.listing_version_id || ""}
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
                variant="compact-outline"
                size="compact"
                onClick={() => setIsRejectDialogOpen(false)}
              >
                Cancel
              </Button>
              <Button
                type="submit"
                variant="compact-destructive"
                size="compact"
              >
                {isApproved ? "Revoke" : "Reject"}
              </Button>
            </DialogFooter>
          </form>
        </DialogContent>
      </Dialog>
    </>
  );
}
