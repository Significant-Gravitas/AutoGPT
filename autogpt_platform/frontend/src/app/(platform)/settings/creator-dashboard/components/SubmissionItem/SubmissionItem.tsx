"use client";

import Image from "next/image";
import { isLocalStoreMediaUrl } from "@/lib/store-media";
import Link from "next/link";
import { motion, useReducedMotion } from "framer-motion";
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
import {
  Delete02Icon,
  EyeIcon,
  ImageNotFound01Icon,
  LinkSquare01Icon,
  MoreVerticalIcon,
  PencilIcon,
} from "@hugeicons/core-free-icons";
import { Icon as UIIcon } from "@/components/atoms/Icon/Icon";

interface EditPayload extends StoreSubmissionEditRequest {
  store_listing_version_id: string;
  graph_id: string;
}

interface Props {
  submission: StoreSubmission;
  rowIndex?: number;
  onView: (submission: StoreSubmission) => void;
  onEdit: (payload: EditPayload) => void;
  onDelete: (submissionId: string) => Promise<void>;
  creatorUsername?: string;
}

const ROW_EASE = [0.16, 1, 0.3, 1] as const;
const ROW_STAGGER = 0.025;
const ROW_STAGGER_CAP = 6;
const ROW_DURATION = 0.22;

export function SubmissionItem({
  submission,
  rowIndex = 0,
  onView,
  onEdit,
  onDelete,
  creatorUsername,
}: Props) {
  const reduceMotion = useReducedMotion();
  const {
    canModify,
    marketplaceUrl,
    handleView,
    handleEdit,
    confirmDeleteOpen,
    setConfirmDeleteOpen,
    isDeleting,
    handleConfirmDelete,
  } = useSubmissionItem({
    submission,
    onView,
    onEdit,
    onDelete,
    creatorUsername,
  });

  const visual = getStatusVisual(submission.status);
  const thumbnail = submission.image_urls?.[0];

  return (
    <motion.tr
      data-testid="submission-row"
      data-agent-id={submission.graph_id}
      data-submission-id={submission.listing_version_id}
      initial={reduceMotion ? false : { opacity: 0, y: 4 }}
      animate={reduceMotion ? undefined : { opacity: 1, y: 0 }}
      transition={
        reduceMotion
          ? undefined
          : {
              duration: ROW_DURATION,
              ease: ROW_EASE,
              delay: Math.min(rowIndex, ROW_STAGGER_CAP) * ROW_STAGGER,
            }
      }
      className="border-b border-separator transition-colors duration-150 last:border-b-0 hover:bg-surface-muted/60"
    >
      <td className="px-4 py-3 align-middle">
        <div className="flex items-center gap-3">
          <div className="relative aspect-video w-20 shrink-0 overflow-hidden rounded-small bg-surface-hover select-none">
            {thumbnail ? (
              <Image
                src={thumbnail}
                unoptimized={isLocalStoreMediaUrl(thumbnail)}
                alt={submission.name}
                fill
                sizes="80px"
                draggable={false}
                className="pointer-events-none object-cover"
              />
            ) : (
              <div className="flex h-full w-full items-center justify-center text-foreground-subtle">
                <UIIcon
                  icon={ImageNotFound01Icon}
                  size={20}
                  className="pointer-events-none"
                />
              </div>
            )}
          </div>
          <div className="flex min-w-0 flex-col">
            <div className="flex min-w-0 items-center gap-2">
              {marketplaceUrl ? (
                <Link
                  href={marketplaceUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex min-w-0 items-center gap-1 truncate text-foreground-base hover:underline"
                  data-testid="submission-marketplace-link"
                >
                  <Text variant="body-medium" as="span" className="truncate">
                    {submission.name}
                  </Text>
                  <UIIcon
                    icon={LinkSquare01Icon}
                    size={14}
                    // oxlint-disable-next-line design-system/no-restyle -- the link glyph is quieter than the name it sits beside
                    className="shrink-0 text-foreground-tertiary"
                    aria-hidden
                  />
                </Link>
              ) : (
                <Text variant="body-medium" as="span" className="truncate">
                  {submission.name}
                </Text>
              )}
              <Text
                variant="small"
                as="span"
                tone="secondary"
                className="shrink-0"
              >
                v{submission.graph_version}
              </Text>
            </div>
            <Text
              variant="small"
              as="span"
              tone="muted"
              className="line-clamp-1 max-w-[420px]"
            >
              {submission.description}
            </Text>
          </div>
        </div>
      </td>

      <td className="px-4 py-3 align-middle">
        <span
          className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium whitespace-nowrap ${visual.pillClass}`}
        >
          <UIIcon icon={visual.Icon} size={12} />
          {visual.label}
        </span>
      </td>

      <td className="px-4 py-3 align-middle text-sm whitespace-nowrap text-foreground-body">
        {formatSubmittedAt(submission.submitted_at)}
      </td>

      <td className="px-4 py-3 text-right align-middle text-sm whitespace-nowrap text-foreground-body tabular-nums">
        {formatRuns(submission.run_count ?? 0)}
      </td>

      <td className="px-2 py-3 text-right align-middle">
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <button
              type="button"
              aria-label="Submission actions"
              data-testid="submission-actions"
              className="inline-flex h-8 w-8 items-center justify-center rounded-full text-foreground-tertiary transition-[background-color,color,transform] duration-150 hover:bg-surface-hover hover:text-foreground-strong active:scale-[0.92] motion-reduce:transition-none motion-reduce:active:scale-100"
            >
              <UIIcon icon={MoreVerticalIcon} size={18} />
            </button>
          </DropdownMenuTrigger>
          <DropdownMenuContent
            align="end"
            sideOffset={6}
            // oxlint-disable-next-line design-system/no-restyle -- the row menu keeps its own open and close timing
            className="min-w-[160px] origin-[var(--radix-dropdown-menu-content-transform-origin)] data-[state=closed]:duration-150 data-[state=open]:duration-200 motion-reduce:!duration-100"
          >
            {canModify ? (
              <DropdownMenuItem onSelect={handleEdit}>
                <span className="flex items-center gap-2">
                  <UIIcon icon={PencilIcon} size={14} />
                  Edit details
                </span>
              </DropdownMenuItem>
            ) : (
              <DropdownMenuItem onSelect={handleView}>
                <span className="flex items-center gap-2">
                  <UIIcon icon={EyeIcon} size={14} />
                  View submission
                </span>
              </DropdownMenuItem>
            )}
            {marketplaceUrl ? (
              <DropdownMenuItem asChild>
                <Link
                  href={marketplaceUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex cursor-pointer items-center gap-2"
                  data-testid="submission-marketplace-menu-link"
                >
                  <UIIcon icon={LinkSquare01Icon} size={14} />
                  View on marketplace
                </Link>
              </DropdownMenuItem>
            ) : null}
            {canModify ? (
              <>
                <DropdownMenuSeparator />
                <DropdownMenuItem
                  onSelect={() => setConfirmDeleteOpen(true)}
                  // oxlint-disable-next-line design-system/no-raw-colors, design-system/no-restyle -- rose delete text kept until the rose -> error hue change is approved
                  className="text-rose-600 focus:text-rose-700"
                >
                  <span className="flex items-center gap-2">
                    <UIIcon icon={Delete02Icon} size={14} />
                    Delete
                  </span>
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
            <Text variant="body" tone="body">
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
    </motion.tr>
  );
}
