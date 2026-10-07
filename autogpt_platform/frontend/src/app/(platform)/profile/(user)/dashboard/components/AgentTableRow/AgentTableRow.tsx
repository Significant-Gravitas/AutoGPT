"use client";

import Image from "next/image";
import { isLocalStoreMediaUrl } from "@/lib/store-media";
import { Text } from "@/components/atoms/Text/Text";

import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/molecules/DropdownMenu/DropdownMenu";
import { Status } from "@/components/__legacy__/Status";
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
      className="hidden items-center border-b border-zinc-300 px-4 py-4 hover:bg-zinc-50 md:flex"
    >
      <div className="grid w-full grid-cols-[minmax(400px,1fr)_180px_140px_100px_100px_40px] items-center gap-4">
        {/* Agent info column */}
        <div className="flex items-center gap-4">
          {showImage && image_urls?.[0] ? (
            <div className="relative aspect-video w-32 shrink-0 overflow-hidden rounded-[10px] bg-zinc-100">
              <Image
                src={image_urls[0]}
                unoptimized={isLocalStoreMediaUrl(image_urls[0])}
                alt={agentName}
                fill
                style={{ objectFit: "cover" }}
                onError={handleImageError}
              />
            </div>
          ) : (
            <div className="flex aspect-video w-32 shrink-0 items-center justify-center overflow-hidden rounded-[10px] bg-zinc-100">
              <Icon
                icon={ImageNotFound01Icon}
                className="h-8 w-8 text-zinc-800"
              />
            </div>
          )}
          <div className="flex flex-col">
            <div className="flex items-center gap-2">
              <Text
                variant="h3"
                size="large-medium"
                tone="primary"
                className="line-clamp-1 text-ellipsis"
                unmask={false}
              >
                {agentName}
              </Text>
              <Text variant="small" tone="muted">
                v{graph_version}
              </Text>
            </div>
            <Text
              variant="body"
              tone="secondary"
              className="line-clamp-1 text-ellipsis"
              unmask={false}
            >
              {description}
            </Text>
          </div>
        </div>

        {/* Date column */}
        <div className="text-sm text-zinc-600">
          {submitted_at && submitted_at.toLocaleDateString()}
        </div>

        {/* Status column */}
        <div data-testid="agent-status">
          <Status status={status} />
        </div>

        {/* Runs column */}
        <div className="text-right text-sm text-zinc-600">
          {run_count?.toLocaleString() ?? "0"}
        </div>

        {/* Reviews column */}
        <div className="text-right">
          {review_avg_rating ? (
            <div className="flex items-center justify-end gap-1">
              <Text variant="body-medium" as="span">
                {review_avg_rating.toFixed(1)}
              </Text>
              <Icon icon={StarIcon} className="h-2 w-2" />
            </div>
          ) : (
            <Text variant="body" as="span" tone="secondary">
              No reviews
            </Text>
          )}
        </div>

        {/* Actions - Three dots menu */}
        <div className="flex justify-end">
          <DropdownMenu>
            <DropdownMenuTrigger data-testid="agent-table-row-actions">
              <Icon
                icon={MoreVerticalIcon}
                size={20}
                className="text-zinc-800"
              />
            </DropdownMenuTrigger>
            <DropdownMenuContent className="rounded-xl p-1 shadow-md">
              {canModify ? (
                <DropdownMenuItem
                  onSelect={handleEdit}
                  className="flex cursor-pointer items-center rounded-md px-3 py-2 hover:bg-zinc-100"
                >
                  <Icon icon={PencilIcon} size={16} className="mr-2" />
                  <span>Edit</span>
                </DropdownMenuItem>
              ) : (
                <DropdownMenuItem
                  onSelect={handleView}
                  className="flex cursor-pointer items-center rounded-md px-3 py-2 hover:bg-zinc-100"
                >
                  <Icon icon={EyeIcon} size={16} className="mr-2" />
                  <span>View</span>
                </DropdownMenuItem>
              )}
              {canModify && (
                <>
                  <DropdownMenuSeparator className="mx-0 my-1 h-px bg-zinc-300" />
                  <DropdownMenuItem
                    onSelect={handleDelete}
                    className="flex cursor-pointer items-center rounded-md px-3 py-2 text-red-500 hover:bg-zinc-100"
                  >
                    <Icon
                      icon={Delete02Icon}
                      size={16}
                      className="mr-2 text-red-500"
                    />
                    <span>Delete</span>
                  </DropdownMenuItem>
                </>
              )}
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>
    </div>
  );
};
