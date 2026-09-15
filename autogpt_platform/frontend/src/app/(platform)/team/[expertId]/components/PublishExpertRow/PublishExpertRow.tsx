"use client";

import type { Expert } from "@/app/api/__generated__/models/expert";
import { Badge } from "@/components/atoms/Badge/Badge";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { ExpertReviewDialog } from "@/components/contextual/ExpertReviewDialog/ExpertReviewDialog";
import { usePublishExpertRow } from "./usePublishExpertRow";

interface Props {
  expert: Expert;
  /** ``EXPERT_PORTABILITY`` and an admin session. The route is admin-only, so
   *  everyone else would only ever get a 403 out of this button. */
  enabled: boolean;
}

export function PublishExpertRow({ expert, enabled }: Props) {
  const {
    isLive,
    isOpen,
    isPublishing,
    preview,
    openDialog,
    closeDialog,
    confirmPublish,
  } = usePublishExpertRow({ expert, enabled });

  if (!enabled) return null;

  return (
    <section className="flex flex-col gap-3 rounded-xl border border-zinc-200 bg-white p-4 sm:flex-row sm:items-center sm:justify-between">
      <div className="min-w-0">
        <div className="flex items-center gap-2">
          <Text variant="large-medium" tone="primary">
            Publish to marketplace
          </Text>
          {isLive ? <Badge variant="success">Live on marketplace</Badge> : null}
        </div>
        <Text variant="small" tone="secondary" className="mt-1 max-w-prose">
          {`Share ${expert.name} with everyone. Skills, soul, workflows and schedules go in; memory, chats and files never do.`}
        </Text>
      </div>
      <Button
        variant="secondary"
        size="small"
        onClick={openDialog}
        data-testid="expert-publish-button"
      >
        {isLive ? "Publish again" : "Publish"}
      </Button>

      <ExpertReviewDialog
        mode="publish"
        open={isOpen}
        preview={preview}
        isSubmitting={isPublishing}
        onClose={closeDialog}
        onConfirm={confirmPublish}
      />
    </section>
  );
}
