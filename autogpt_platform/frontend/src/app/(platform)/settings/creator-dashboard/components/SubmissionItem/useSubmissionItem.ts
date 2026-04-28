import { useState } from "react";
import * as Sentry from "@sentry/nextjs";

import { SubmissionStatus } from "@/app/api/__generated__/models/submissionStatus";
import type { StoreSubmission } from "@/app/api/__generated__/models/storeSubmission";
import type { StoreSubmissionEditRequest } from "@/app/api/__generated__/models/storeSubmissionEditRequest";
import { toast } from "@/components/molecules/Toast/use-toast";

interface EditPayload extends StoreSubmissionEditRequest {
  store_listing_version_id: string | undefined;
  graph_id: string;
}

interface Args {
  submission: StoreSubmission;
  onView: (submission: StoreSubmission) => void;
  onEdit: (payload: EditPayload) => void;
  onDelete: (submissionId: string) => Promise<void>;
}

export function useSubmissionItem({
  submission,
  onView,
  onEdit,
  onDelete,
}: Args) {
  const [confirmDeleteOpen, setConfirmDeleteOpen] = useState(false);
  const [isDeleting, setIsDeleting] = useState(false);

  const canModify = submission.status === SubmissionStatus.PENDING;

  function handleView() {
    onView(submission);
  }

  function handleEdit() {
    onEdit({
      name: submission.name,
      sub_heading: submission.sub_heading,
      description: submission.description,
      image_urls: submission.image_urls,
      video_url: submission.video_url,
      categories: submission.categories,
      changes_summary: submission.changes_summary || "Update Submission",
      store_listing_version_id: submission.listing_version_id,
      graph_id: submission.graph_id,
    });
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
    handleView,
    handleEdit,
    confirmDeleteOpen,
    setConfirmDeleteOpen,
    isDeleting,
    handleConfirmDelete,
  };
}
