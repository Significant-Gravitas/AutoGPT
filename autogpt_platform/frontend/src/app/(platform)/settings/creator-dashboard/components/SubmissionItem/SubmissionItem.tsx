"use client";

import Image from "next/image";
import {
  CheckSquareIcon,
  DotsThreeVerticalIcon,
  EyeIcon,
  ImageBrokenIcon,
  PencilSimpleIcon,
  SquareIcon,
  StarIcon,
  TrashIcon,
} from "@phosphor-icons/react";

import type { StoreSubmission } from "@/app/api/__generated__/models/storeSubmission";
import type { StoreSubmissionEditRequest } from "@/app/api/__generated__/models/storeSubmissionEditRequest";
import { Button } from "@/components/atoms/Button/Button";
import { Text } from "@/components/atoms/Text/Text";
import { Dialog } from "@/components/molecules/Dialog/Dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/molecules/DropdownMenu/DropdownMenu";

import { formatRuns, formatSubmittedAt, getStatusVisual } from "../../helpers";
import { useSubmissionItem } from "./useSubmissionItem";

interface EditPayload extends StoreSubmissionEditRequest {
  store_listing_version_id: string | undefined;
  graph_id: string;
}

interface Props {
  submission: StoreSubmission;
  selected: boolean;
  onToggleSelected: () => void;
  onView: (submission: StoreSubmission) => void;
  onEdit: (payload: EditPayload) => void;
  onDelete: (submissionId: string) => Promise<void>;
}

export function SubmissionItem({
  submission,
  selected,
  onToggleSelected,
  onView,
  onEdit,
  onDelete,
}: Props) {
  const {
    canModify,
    handleView,
    handleEdit,
    confirmDeleteOpen,
    setConfirmDeleteOpen,
    isDeleting,
    handleConfirmDelete,
  } = useSubmissionItem({ submission, onView, onEdit, onDelete });

  const visual = getStatusVisual(submission.status);
  const StatusIcon = visual.Icon;
  const thumbnail = submission.image_urls?.[0];

  return (
    <tr
      data-testid="submission-row"
      data-agent-id={submission.graph_id}
      data-submission-id={submission.listing_version_id}
      data-selected={selected}
      className="ease-[cubic-bezier(0.16,1,0.3,1)] border-b border-zinc-100 transition-colors duration-150 last:border-b-0 data-[selected=true]:bg-zinc-100 hover:bg-zinc-50/60"
    >
      <td className="w-[48px] px-3 py-3 align-middle">
        {canModify ? (
          <button
            type="button"
            role="checkbox"
            aria-checked={selected}
            aria-label={`Select ${submission.name}`}
            onClick={onToggleSelected}
            className={`shrink-0 transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-zinc-800 ${
              selected
                ? "text-zinc-800 hover:text-zinc-900"
                : "text-zinc-500 hover:text-zinc-700"
            }`}
          >
            {selected ? (
              <CheckSquareIcon size={20} weight="fill" />
            ) : (
              <SquareIcon size={20} />
            )}
          </button>
        ) : null}
      </td>
      <td className="px-4 py-3 align-middle">
        <div className="flex items-center gap-3">
          <div className="relative aspect-video w-20 shrink-0 overflow-hidden rounded-[8px] bg-zinc-100">
            {thumbnail ? (
              <Image
                src={thumbnail}
                alt={submission.name}
                fill
                sizes="80px"
                style={{ objectFit: "cover" }}
              />
            ) : (
              <div className="flex h-full w-full items-center justify-center">
                <ImageBrokenIcon size={20} className="text-zinc-400" />
              </div>
            )}
          </div>
          <div className="flex min-w-0 flex-col">
            <div className="flex min-w-0 items-center gap-2">
              <Text
                variant="body-medium"
                as="span"
                className="truncate text-textBlack"
              >
                {submission.name}
              </Text>
              <Text
                variant="small"
                as="span"
                className="shrink-0 text-zinc-600"
              >
                v{submission.graph_version}
              </Text>
            </div>
            <Text
              variant="small"
              as="span"
              className="line-clamp-1 max-w-[420px] text-zinc-500"
            >
              {submission.description}
            </Text>
          </div>
        </div>
      </td>

      <td className="px-4 py-3 align-middle">
        <span
          className={`inline-flex items-center gap-1.5 whitespace-nowrap rounded-full px-2.5 py-1 text-xs font-medium ${visual.pillClass}`}
        >
          <StatusIcon size={12} weight="fill" />
          {visual.label}
        </span>
      </td>

      <td className="whitespace-nowrap px-4 py-3 align-middle text-sm text-zinc-700">
        {formatSubmittedAt(submission.submitted_at)}
      </td>

      <td className="whitespace-nowrap px-4 py-3 text-right align-middle text-sm tabular-nums text-zinc-700">
        {formatRuns(submission.run_count ?? 0)}
      </td>

      <td className="whitespace-nowrap px-4 py-3 text-right align-middle text-sm text-zinc-700">
        {submission.review_avg_rating && submission.review_avg_rating > 0 ? (
          <span className="inline-flex items-center justify-end gap-1 tabular-nums">
            {submission.review_avg_rating.toFixed(1)}
            <StarIcon size={12} weight="fill" className="text-amber-500" />
          </span>
        ) : (
          <span className="text-zinc-400">—</span>
        )}
      </td>

      <td className="px-2 py-3 text-right align-middle">
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <button
              type="button"
              aria-label="Submission actions"
              data-testid="submission-actions"
              className="ease-[cubic-bezier(0.16,1,0.3,1)] inline-flex h-8 w-8 items-center justify-center rounded-full text-zinc-500 transition-[background-color,color,transform] duration-150 hover:bg-zinc-100 hover:text-zinc-900 active:scale-[0.92] motion-reduce:transition-none motion-reduce:active:scale-100"
            >
              <DotsThreeVerticalIcon size={18} weight="bold" />
            </button>
          </DropdownMenuTrigger>
          <DropdownMenuContent
            align="end"
            sideOffset={6}
            className="data-[state=open]:ease-[cubic-bezier(0.16,1,0.3,1)] data-[state=closed]:ease-[cubic-bezier(0.4,0,1,1)] min-w-[160px] origin-[var(--radix-dropdown-menu-content-transform-origin)] data-[state=closed]:duration-150 data-[state=open]:duration-200 motion-reduce:!duration-100"
          >
            {canModify ? (
              <DropdownMenuItem
                onSelect={handleEdit}
                className="flex cursor-pointer items-center gap-2"
              >
                <PencilSimpleIcon size={14} />
                Edit details
              </DropdownMenuItem>
            ) : (
              <DropdownMenuItem
                onSelect={handleView}
                className="flex cursor-pointer items-center gap-2"
              >
                <EyeIcon size={14} />
                View submission
              </DropdownMenuItem>
            )}
            {canModify ? (
              <>
                <DropdownMenuSeparator />
                <DropdownMenuItem
                  onSelect={() => setConfirmDeleteOpen(true)}
                  className="flex cursor-pointer items-center gap-2 text-rose-600 focus:text-rose-700"
                >
                  <TrashIcon size={14} />
                  Delete
                </DropdownMenuItem>
              </>
            ) : null}
          </DropdownMenuContent>
        </DropdownMenu>

        <Dialog
          title="Delete submission?"
          styling={{ maxWidth: "420px" }}
          controlled={{
            isOpen: confirmDeleteOpen,
            set: (open) => setConfirmDeleteOpen(open),
          }}
        >
          <Dialog.Content>
            <Text variant="body" className="text-zinc-700">
              This will remove <strong>{submission.name}</strong> from the
              store. This action cannot be undone.
            </Text>
            <Dialog.Footer>
              <Button
                variant="ghost"
                size="small"
                onClick={() => setConfirmDeleteOpen(false)}
                disabled={isDeleting}
              >
                Cancel
              </Button>
              <Button
                variant="destructive"
                size="small"
                onClick={handleConfirmDelete}
                loading={isDeleting}
              >
                {isDeleting ? "Deleting" : "Delete submission"}
              </Button>
            </Dialog.Footer>
          </Dialog.Content>
        </Dialog>
      </td>
    </tr>
  );
}
