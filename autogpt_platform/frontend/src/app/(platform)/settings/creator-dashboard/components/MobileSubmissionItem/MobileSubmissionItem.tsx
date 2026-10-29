"use client";

import Image from "next/image";
import { isLocalStoreMediaUrl } from "@/lib/store-media";
import Link from "next/link";
import { motion, useReducedMotion } from "framer-motion";
import type { StoreSubmission } from "@/app/api/__generated__/models/storeSubmission";
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

import type { EditPayload } from "../../helpers";
import { formatRuns, formatSubmittedAt, getStatusVisual } from "../../helpers";
import { useSubmissionItem } from "../SubmissionItem/useSubmissionItem";
import {
  Delete02Icon,
  EyeIcon,
  ImageNotFound01Icon,
  LinkSquare01Icon,
  MoreVerticalIcon,
  PencilIcon,
  StarIcon,
} from "@hugeicons/core-free-icons";
import { Icon as UIIcon } from "@/components/atoms/Icon/Icon";

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

export function MobileSubmissionItem({
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
  const hasRating =
    !!submission.review_avg_rating && submission.review_avg_rating > 0;

  return (
    <motion.article
      data-testid="submission-card"
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
      className="flex flex-col gap-3 border-b border-separator px-3 py-3 last:border-b-0"
    >
      <div className="flex items-start gap-3">
        <div className="relative aspect-video w-16 shrink-0 select-none overflow-hidden rounded-small bg-surface-hover">
          {thumbnail ? (
            <Image
              src={thumbnail}
              unoptimized={isLocalStoreMediaUrl(thumbnail)}
              alt={submission.name}
              fill
              sizes="64px"
              draggable={false}
              className="pointer-events-none object-cover"
            />
          ) : (
            <div className="flex h-full w-full items-center justify-center text-foreground-subtle">
              <UIIcon
                icon={ImageNotFound01Icon}
                size={18}
                className="pointer-events-none"
              />
            </div>
          )}
        </div>

        <div className="flex min-w-0 flex-1 flex-col">
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
          {submission.description ? (
            <Text
              variant="small"
              as="span"
              tone="muted"
              className="mt-0.5 line-clamp-2 break-words"
            >
              {submission.description}
            </Text>
          ) : null}
        </div>

        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <button
              type="button"
              aria-label="Submission actions"
              data-testid="submission-actions"
              className="-mr-1 inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-foreground-tertiary transition-colors hover:bg-surface-hover hover:text-foreground-strong active:scale-[0.92] motion-reduce:active:scale-100"
            >
              <UIIcon icon={MoreVerticalIcon} size={18} />
            </button>
          </DropdownMenuTrigger>
          <DropdownMenuContent
            align="end"
            sideOffset={6}
            className="min-w-[160px]"
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
      </div>

      <div className="flex flex-wrap items-center gap-x-2 gap-y-1.5">
        <span
          className={`inline-flex items-center gap-1 whitespace-nowrap rounded-full px-2 py-0.5 text-caption-sm font-medium leading-inherit ${visual.pillClass}`}
        >
          <UIIcon icon={visual.Icon} size={10} />
          {visual.label}
        </span>
        <span className="whitespace-nowrap text-xs text-foreground-tertiary">
          {formatSubmittedAt(submission.submitted_at)}
        </span>
        <span aria-hidden className="text-xs text-foreground-disabled">
          ·
        </span>
        <span className="whitespace-nowrap text-xs tabular-nums text-foreground-tertiary">
          {formatRuns(submission.run_count ?? 0)} runs
        </span>
        {hasRating ? (
          <>
            <span aria-hidden className="text-xs text-foreground-disabled">
              ·
            </span>
            <span className="inline-flex items-center gap-1 whitespace-nowrap text-xs tabular-nums text-foreground-tertiary">
              {submission.review_avg_rating!.toFixed(1)}
              {/* oxlint-disable-next-line design-system/no-restyle -- the star is amber beside grey run counts */}
              <UIIcon icon={StarIcon} size={10} className="text-warning" />
            </span>
          </>
        ) : null}
      </div>

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
            This will remove <strong>{submission.name}</strong> from the store.
            This action cannot be undone.
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
    </motion.article>
  );
}
