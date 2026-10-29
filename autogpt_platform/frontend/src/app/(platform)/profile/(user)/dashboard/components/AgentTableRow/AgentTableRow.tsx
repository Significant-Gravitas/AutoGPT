"use client";

import Image from "next/image";
import { isLocalStoreMediaUrl } from "@/lib/store-media";
import { Text } from "@/components/atoms/Text/Text";

import * as DropdownMenu from "@radix-ui/react-dropdown-menu";
import { useAgentTableRow } from "./useAgentTableRow";
import { StoreSubmission } from "@/app/api/__generated__/models/storeSubmission";
import { SubmissionStatus } from "@/app/api/__generated__/models/submissionStatus";
import { StoreSubmissionEditRequest } from "@/app/api/__generated__/models/storeSubmissionEditRequest";
import {
  Delete02Icon,
  EyeIcon,
  ImageNotFound01Icon,
  MoreVerticalIcon,
  PencilIcon,
  StarIcon,
} from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";
import { useImageFallback } from "@/hooks/useImageFallback";
import { SubmissionStatus as SubmissionStatusBadge } from "../SubmissionStatus/SubmissionStatus";

export type AgentTableRowProps = {
  storeAgentSubmission: StoreSubmission;
  onViewSubmission: (submission: StoreSubmission) => void;
  onDeleteSubmission: (submission_id: string) => void;
  onEditSubmission: (
    submission: StoreSubmissionEditRequest & {
      store_listing_version_id: string | undefined;
      graph_id: string;
    },
  ) => void;
};

export const AgentTableRow = ({
  storeAgentSubmission,
  onViewSubmission,
  onDeleteSubmission,
  onEditSubmission,
}: AgentTableRowProps) => {
  const { handleView, handleDelete, handleEdit } = useAgentTableRow({
    storeAgentSubmission,
    onViewSubmission,
    onDeleteSubmission,
    onEditSubmission,
  });

  const {
    listing_version_id,
    graph_id,
    graph_version,
    name: agentName,
    description,
    image_urls,
    submitted_at,
    status,
    run_count,
    review_avg_rating,
  } = storeAgentSubmission;

  const { showImage, handleImageError } = useImageFallback(image_urls?.[0]);
  const canModify = status === SubmissionStatus.PENDING;

  return (
    <div
      data-testid="agent-table-row"
      data-agent-id={graph_id}
      data-submission-id={listing_version_id}
      className="hidden items-center border-b border-border-strong px-4 py-4 hover:bg-background md:flex"
    >
      <div className="grid w-full grid-cols-[minmax(400px,1fr)_180px_140px_100px_100px_40px] items-center gap-4">
        {/* Agent info column */}
        <div className="flex items-center gap-4">
          {showImage && image_urls?.[0] ? (
            <div className="relative aspect-video w-32 shrink-0 overflow-hidden rounded-lg bg-surface-hover">
              <Image
                src={image_urls[0]}
                unoptimized={isLocalStoreMediaUrl(image_urls[0])}
                alt={agentName}
                fill
                className="object-cover"
                onError={handleImageError}
              />
            </div>
          ) : (
            <div className="flex aspect-video w-32 shrink-0 items-center justify-center overflow-hidden rounded-lg bg-surface-hover text-foreground-emphasis">
              <Icon icon={ImageNotFound01Icon} className="h-8 w-8" />
            </div>
          )}
          <div className="flex flex-col">
            <div className="flex items-center gap-2">
              <Text
                variant="h3"
                size="large-medium"
                tone="emphasis"
                className="line-clamp-1"
              >
                {agentName}
              </Text>
              <Text variant="body" size="small" tone="muted">
                v{graph_version}
              </Text>
            </div>
            <Text variant="body" tone="secondary" className="line-clamp-1">
              {description}
            </Text>
          </div>
        </div>

        {/* Date column */}
        <div className="text-sm text-foreground-secondary">
          {submitted_at && submitted_at.toLocaleDateString()}
        </div>

        {/* Status column */}
        <div data-testid="agent-status">
          <SubmissionStatusBadge status={status} />
        </div>

        {/* Runs column */}
        <div className="text-right text-sm text-foreground-secondary">
          {run_count?.toLocaleString() ?? "0"}
        </div>

        {/* Reviews column */}
        <div className="text-right">
          {review_avg_rating ? (
            <div className="flex items-center justify-end gap-1">
              <span className="text-sm font-medium">
                {review_avg_rating.toFixed(1)}
              </span>
              <Icon icon={StarIcon} className="h-2 w-2" />
            </div>
          ) : (
            <span className="text-sm text-foreground-secondary">
              No reviews
            </span>
          )}
        </div>

        {/* Actions - Three dots menu */}
        <div className="flex justify-end">
          <DropdownMenu.Root>
            <DropdownMenu.Trigger
              data-testid="agent-table-row-actions"
              className="text-foreground-emphasis"
            >
              <Icon icon={MoreVerticalIcon} className="h-5 w-5" />
            </DropdownMenu.Trigger>
            <DropdownMenu.Content className="z-10 rounded-xl border bg-surface p-1 shadow-md">
              {canModify ? (
                <DropdownMenu.Item
                  onSelect={handleEdit}
                  className="flex cursor-pointer items-center rounded-md px-3 py-2 hover:bg-surface-hover"
                >
                  <Icon icon={PencilIcon} className="mr-2 h-4 w-4" />
                  <span>Edit</span>
                </DropdownMenu.Item>
              ) : (
                <DropdownMenu.Item
                  onSelect={handleView}
                  className="flex cursor-pointer items-center rounded-md px-3 py-2 hover:bg-surface-hover"
                >
                  <Icon icon={EyeIcon} className="mr-2 h-4 w-4" />
                  <span>View</span>
                </DropdownMenu.Item>
              )}
              {canModify && (
                <>
                  <DropdownMenu.Separator className="my-1 h-px bg-border-strong" />
                  <DropdownMenu.Item
                    onSelect={handleDelete}
                    className="flex cursor-pointer items-center rounded-md px-3 py-2 text-error hover:bg-surface-hover"
                  >
                    <Icon icon={Delete02Icon} className="mr-2 h-4 w-4" />
                    <span>Delete</span>
                  </DropdownMenu.Item>
                </>
              )}
            </DropdownMenu.Content>
          </DropdownMenu.Root>
        </div>
      </div>
    </div>
  );
};
