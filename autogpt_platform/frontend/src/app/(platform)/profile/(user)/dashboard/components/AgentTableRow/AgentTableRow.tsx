"use client";

import Image from "next/image";
import { Text } from "@/components/atoms/Text/Text";

import * as DropdownMenu from "@radix-ui/react-dropdown-menu";
import { Status } from "@/components/__legacy__/Status";
import { useAgentTableRow } from "./useAgentTableRow";
import { StoreSubmission } from "@/app/api/__generated__/models/storeSubmission";
import {
  DotsThreeVerticalIcon,
  EyeIcon,
  ImageBroken,
  StarIcon,
  TrashIcon,
  PencilIcon,
} from "@phosphor-icons/react/dist/ssr";
import { SubmissionStatus } from "@/app/api/__generated__/models/submissionStatus";
import { StoreSubmissionEditRequest } from "@/app/api/__generated__/models/storeSubmissionEditRequest";

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

  const canModify = status === SubmissionStatus.PENDING;

  return (
    <div
      data-testid="agent-table-row"
      data-agent-id={graph_id}
      data-submission-id={listing_version_id}
      className="hidden items-center border-b border-neutral-300 px-4 py-4 hover:bg-neutral-50 dark:border-neutral-700 dark:hover:bg-neutral-800 md:flex"
    >
      <div className="grid w-full grid-cols-[minmax(400px,1fr),180px,140px,100px,100px,40px] items-center gap-4">
        {/* Agent info column */}
        <div className="flex items-center gap-4">
          {image_urls?.[0] ? (
            <div className="relative aspect-video w-32 shrink-0 overflow-hidden rounded-[10px] bg-zinc-100">
              <Image
                src={image_urls?.[0] ?? ""}
                alt={agentName}
                fill
                style={{ objectFit: "cover" }}
              />
            </div>
          ) : (
            <div className="flex aspect-video w-32 shrink-0 items-center justify-center overflow-hidden rounded-[10px] bg-zinc-100">
              <ImageBroken className="h-8 w-8 text-zinc-800" />
            </div>
          )}
          <div className="flex flex-col">
            <div className="flex items-center gap-2">
              <Text
                variant="h3"
                className="line-clamp-1 text-ellipsis text-neutral-800 dark:text-neutral-200"
                size="large-medium"
              >
                {agentName}
              </Text>
              <Text
                variant="body"
                size="small"
                className="text-neutral-500 dark:text-neutral-400"
              >
                v{graph_version}
              </Text>
            </div>
            <Text
              variant="body"
              className="line-clamp-1 text-ellipsis text-neutral-600 dark:text-neutral-400"
            >
              {description}
            </Text>
          </div>
        </div>

        {/* Date column */}
        <div className="text-sm text-neutral-600 dark:text-neutral-400">
          {submitted_at && submitted_at.toLocaleDateString()}
        </div>

        {/* Status column */}
        <div data-testid="agent-status">
          <Status status={status} />
        </div>

        {/* Runs column */}
        <div className="text-right text-sm text-neutral-600 dark:text-neutral-400">
          {run_count?.toLocaleString() ?? "0"}
        </div>

        {/* Reviews column */}
        <div className="text-right">
          {review_avg_rating ? (
            <div className="flex items-center justify-end gap-1">
              <span className="text-sm font-medium">
                {review_avg_rating.toFixed(1)}
              </span>
              <StarIcon weight="fill" className="h-2 w-2" />
            </div>
          ) : (
            <span className="text-sm text-neutral-600 dark:text-neutral-400">
              No reviews
            </span>
          )}
        </div>

        {/* Actions - Three dots menu */}
        <div className="flex justify-end">
          <DropdownMenu.Root>
            <DropdownMenu.Trigger data-testid="agent-table-row-actions">
              <DotsThreeVerticalIcon className="h-5 w-5 text-neutral-800" />
            </DropdownMenu.Trigger>
            <DropdownMenu.Content className="z-10 rounded-xl border bg-white p-1 shadow-md dark:bg-gray-800">
              {canModify ? (
                <DropdownMenu.Item
                  onSelect={handleEdit}
                  className="flex cursor-pointer items-center rounded-md px-3 py-2 hover:bg-gray-100 dark:hover:bg-gray-700"
                >
                  <PencilIcon className="mr-2 h-4 w-4 dark:text-gray-100" />
                  <span className="dark:text-gray-100">Edit</span>
                </DropdownMenu.Item>
              ) : (
                <DropdownMenu.Item
                  onSelect={handleView}
                  className="flex cursor-pointer items-center rounded-md px-3 py-2 hover:bg-gray-100 dark:hover:bg-gray-700"
                >
                  <EyeIcon className="mr-2 h-4 w-4 dark:text-gray-100" />
                  <span className="dark:text-gray-100">View</span>
                </DropdownMenu.Item>
              )}
              {canModify && (
                <>
                  <DropdownMenu.Separator className="my-1 h-px bg-gray-300 dark:bg-gray-600" />
                  <DropdownMenu.Item
                    onSelect={handleDelete}
                    className="flex cursor-pointer items-center rounded-md px-3 py-2 text-red-500 hover:bg-gray-100 dark:hover:bg-gray-700"
                  >
                    <TrashIcon className="mr-2 h-4 w-4 text-red-500 dark:text-red-400" />
                    <span className="dark:text-red-400">Delete</span>
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
