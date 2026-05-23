"use client";

import type { CopilotTurnJobInfo } from "@/app/api/__generated__/models/copilotTurnJobInfo";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import { ChatCircleTextIcon, EyeIcon, TrashIcon } from "@phosphor-icons/react";
import Link from "next/link";
import { useFollowupListItem } from "./useFollowupListItem";

interface Props {
  followup: CopilotTurnJobInfo;
}

export function FollowupListItem({ followup }: Props) {
  const {
    sessionHref,
    nextRunLabel,
    nextRunTitle,
    recurrenceLabel,
    messagePreview,
    isDeleteOpen,
    openDelete,
    closeDelete,
    isDeleting,
    handleDelete,
    isViewOpen,
    openView,
    closeView,
    fullMessage,
  } = useFollowupListItem({ followup });

  const sessionLabel = followup.session_id
    ? `Session ${followup.session_id.slice(0, 8)}`
    : "New chat";

  const detailContent = (
    <>
      <div className="flex h-9 w-9 flex-shrink-0 items-center justify-center rounded-large border border-slate-50 bg-yellow-50">
        <ChatCircleTextIcon
          size={18}
          className="text-yellow-700"
          weight="bold"
        />
      </div>
      <div className="flex min-w-0 flex-col gap-1">
        <Text
          variant="body-medium"
          className="block w-full truncate text-ellipsis"
        >
          {messagePreview}
        </Text>
        <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5">
          <Text variant="small" className="!text-zinc-500" title={nextRunTitle}>
            {nextRunLabel}
          </Text>
          <span className="text-zinc-300">•</span>
          <Text variant="small" className="!text-zinc-500">
            {recurrenceLabel}
          </Text>
          <span className="text-zinc-300">•</span>
          <Text variant="small" className="!text-zinc-400">
            {sessionLabel}
          </Text>
        </div>
      </div>
    </>
  );

  return (
    <div
      className="flex w-full flex-col gap-3 rounded-large border border-zinc-200 bg-white p-4 sm:flex-row sm:items-center sm:justify-between"
      data-testid="followup-row"
      data-followup-id={followup.id}
    >
      {sessionHref ? (
        <Link
          href={sessionHref}
          className="flex min-w-0 flex-1 items-start gap-3 hover:opacity-80"
          data-testid="followup-open-session"
        >
          {detailContent}
        </Link>
      ) : (
        <div
          className="flex min-w-0 flex-1 items-start gap-3"
          data-testid="followup-row-no-session"
        >
          {detailContent}
        </div>
      )}

      <div className="flex flex-shrink-0 items-center gap-2">
        <Button
          variant="secondary"
          size="small"
          onClick={openView}
          data-testid="followup-view-button"
          aria-label="View follow-up"
        >
          <EyeIcon className="mr-1 h-4 w-4" />
          View
        </Button>
        <Button
          variant="secondary"
          size="small"
          onClick={openDelete}
          data-testid="followup-delete-button"
          aria-label="Delete follow-up"
        >
          <TrashIcon className="mr-1 h-4 w-4" />
          Delete
        </Button>
      </div>

      <Dialog
        controlled={{ isOpen: isViewOpen, set: closeView }}
        styling={{ maxWidth: "40rem" }}
        title="Follow-up details"
      >
        <Dialog.Content>
          <div className="flex flex-col gap-3">
            <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5">
              <Text variant="small" className="!text-zinc-500">
                {nextRunLabel}
              </Text>
              <span className="text-zinc-300">•</span>
              <Text variant="small" className="!text-zinc-500">
                {recurrenceLabel}
              </Text>
              <span className="text-zinc-300">•</span>
              <Text variant="small" className="!text-zinc-400">
                {sessionLabel}
              </Text>
            </div>
            <pre
              className="max-h-[60vh] overflow-auto rounded-medium bg-zinc-50 p-3 text-sm text-zinc-800"
              style={{ whiteSpace: "pre-wrap" }}
              data-testid="followup-view-body"
            >
              {fullMessage}
            </pre>
          </div>
        </Dialog.Content>
      </Dialog>

      <Dialog
        controlled={{ isOpen: isDeleteOpen, set: closeDelete }}
        styling={{ maxWidth: "32rem" }}
        title="Delete follow-up"
      >
        <Dialog.Content>
          <div className="flex flex-col gap-4">
            <Text variant="large">
              Delete this scheduled follow-up? The copilot will not send the
              message and you can recreate it from chat if needed.
            </Text>
            <Dialog.Footer>
              <Button
                variant="secondary"
                disabled={isDeleting}
                onClick={() => closeDelete(false)}
              >
                Keep it
              </Button>
              <Button
                variant="destructive"
                onClick={handleDelete}
                loading={isDeleting}
                data-testid="followup-confirm-delete"
              >
                Yes, delete
              </Button>
            </Dialog.Footer>
          </div>
        </Dialog.Content>
      </Dialog>
    </div>
  );
}
