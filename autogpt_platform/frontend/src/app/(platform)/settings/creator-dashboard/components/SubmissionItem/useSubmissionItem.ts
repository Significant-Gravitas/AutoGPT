import { useState } from "react";
import * as Sentry from "@sentry/nextjs";

import { SubmissionStatus } from "@/app/api/__generated__/models/submissionStatus";
import type { StoreSubmission } from "@/app/api/__generated__/models/storeSubmission";
import { toast } from "@/components/molecules/Toast/use-toast";
import { getApprovedMarketplaceUrl } from "@/lib/utils";

import { buildEditPayload, type EditPayload } from "../../helpers";

interface Args {
  submission: StoreSubmission;
  onView: (submission: StoreSubmission) => void;
  onEdit: (payload: EditPayload) => void;
  onDelete: (submissionId: string) => Promise<void>;
  creatorUsername?: string;
}

export function useSubmissionItem({
  submission,
  onView,
  onEdit,
  onDelete,
  creatorUsername,
}: Args) {
  const [confirmDeleteOpen, setConfirmDeleteOpen] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);

  const canModify = submission.status === SubmissionStatus.PENDING;
  const isApproved = submission.status === SubmissionStatus.APPROVED;
  const marketplaceUrl = getApprovedMarketplaceUrl({
    creatorUsername,
    slug: submission.slug,
    isApproved,
  });

  function handleView() {
    onView(submission);
  }

  function handleEdit() {
    onEdit(buildEditPayload(submission));
  }

  async function handleConfirmDelete() {
    setIsDeleting(true);
    try {
      await onDelete(submission.listing_version_id);
      setConfirmDeleteOpen(false);
    } catch (err) {
      Sentry.captureException(err);
      toast({
        title: "Couldn't delete submission",
        description: err instanceof Error ? err.message : undefined,
        variant: "destructive",
      });
    } finally {
      setIsDeleting(false);
    }
  }

  return {
    canModify,
    isApproved,
    marketplaceUrl,
    handleView,
    handleEdit,
    confirmDeleteOpen,
    setConfirmDeleteOpen,
    isDeleting,
    handleConfirmDelete,
  };
}
