"use client";

import { useState } from "react";
import { Button } from "@/components/atoms/Button/Button";
import { Input } from "@/components/atoms/Input/Input";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import {
  CancelCircleIcon,
  CheckmarkCircle02Icon,
} from "@hugeicons/core-free-icons";
import { Label } from "@/components/__legacy__/ui/label";
import { Textarea } from "@/components/__legacy__/ui/textarea";
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
          size="small"
          variant="outline"
          className="text-green-600 hover:bg-green-50 hover:text-green-700"
          leadingIcon={CheckmarkCircle02Icon}
          onClick={(e) => {
            e.stopPropagation();
            setIsApproveDialogOpen(true);
          }}
        >
          Approve
        </Button>
      )}
      <Button
        size="small"
        variant="outline"
        className="text-red-600 hover:bg-red-50 hover:text-red-700"
        leadingIcon={CancelCircleIcon}
        onClick={(e) => {
          e.stopPropagation();
          setIsRejectDialogOpen(true);
        }}
      >
        {isApproved ? "Revoke" : "Reject"}
      </Button>

      {/* Approve Dialog */}
      <Dialog
        title="Approve Agent"
        variant="compact"
        controlled={{
          isOpen: isApproveDialogOpen,
          set: setIsApproveDialogOpen,
        }}
      >
        <Dialog.Content>
          <Text variant="body" tone="muted">
            Are you sure you want to approve this agent? This will make it
            available in the marketplace.
          </Text>

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

            <Dialog.Footer>
              <Button
                type="button"
                variant="outline"
                size="small"
                onClick={() => setIsApproveDialogOpen(false)}
              >
                Cancel
              </Button>
              <Button type="submit" variant="primary" size="small">
                Approve
              </Button>
            </Dialog.Footer>
          </form>
        </Dialog.Content>
      </Dialog>

      {/* Reject Dialog */}
      <Dialog
        title={isApproved ? "Revoke Approved Agent" : "Reject Agent"}
        variant="compact"
        controlled={{
          isOpen: isRejectDialogOpen,
          set: setIsRejectDialogOpen,
        }}
      >
        <Dialog.Content>
          <Text variant="body" tone="muted">
            {isApproved
              ? "Are you sure you want to revoke approval for this agent? This will remove it from the marketplace."
              : "Please provide feedback on why this agent is being rejected."}
          </Text>

          <form action={handleRejectSubmit}>
            <input
              type="hidden"
              name="id"
              value={version.listing_version_id || ""}
            />

            <div className="grid gap-4 py-4">
              <Input
                id="comments"
                type="textarea"
                label="Comments for Creator"
                labelVariant="body-medium"
                size="small"
                wrapperClassName="mb-0"
                name="comments"
                placeholder="Provide feedback for the agent creator"
                required
              />

              <Input
                id="internal_comments"
                type="textarea"
                label="Internal Comments"
                labelVariant="body-medium"
                size="small"
                wrapperClassName="mb-0"
                name="internal_comments"
                placeholder="Add any internal notes (not visible to creator)"
              />
            </div>

            <Dialog.Footer>
              <Button
                type="button"
                variant="outline"
                size="small"
                onClick={() => setIsRejectDialogOpen(false)}
              >
                Cancel
              </Button>
              <Button type="submit" variant="destructive" size="small">
                {isApproved ? "Revoke" : "Reject"}
              </Button>
            </Dialog.Footer>
          </form>
        </Dialog.Content>
      </Dialog>
    </>
  );
}
