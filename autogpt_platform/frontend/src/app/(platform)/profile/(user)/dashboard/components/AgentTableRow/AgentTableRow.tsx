"use client";

import Image from "next/image";
import { Text } from "@/components/atoms/Text/Text";

import * as DropdownMenu from "@radix-ui/react-dropdown-menu";
import { Status, StatusType } from "@/components/agptui/Status";
import { useAgentTableRow } from "./useAgentTableRow";
import { StoreSubmission } from "@/app/api/__generated__/models/storeSubmission";
import {
  DotsThreeVerticalIcon,
  Eye,
  ImageBroken,
  Star,
  Trash,
} from "@phosphor-icons/react/dist/ssr";

export interface AgentTableRowProps {
  agent_id: string;
  agent_version: number;
  agentName: string;
  sub_heading: string;
  description: string;
  imageSrc: string[];
  date_submitted: string;
  status: StatusType;
  runs: number;
  rating: number;
  dateSubmitted: string;
  id: number;
  onViewSubmission: (submission: StoreSubmission) => void;
  onDeleteSubmission: (submission_id: string) => void;
}

export const AgentTableRow = ({
  agent_id,
  agent_version,
  agentName,
  sub_heading,
  description,
  imageSrc,
  dateSubmitted,
  status,
  runs,
  rating,
  id,
  onViewSubmission,
  onDeleteSubmission,
}: AgentTableRowProps) => {
  const { handleView, handleDelete } = useAgentTableRow({
    id,
    onViewSubmission,
    onDeleteSubmission,
    agent_id,
    agent_version,
    agentName,
    sub_heading,
    description,
    imageSrc,
    dateSubmitted,
    status: status.toUpperCase(),
    runs,
    rating,
  });

  return (
    <div
      data-testid="agent-table-row"
      data-agent-name={agentName}
      className="hidden items-center border-b border-neutral-300 px-4 py-4 hover:bg-neutral-50 dark:border-neutral-700 dark:hover:bg-neutral-800 md:flex"
    >
      <div className="grid w-full grid-cols-[minmax(400px,1fr),180px,140px,100px,100px,40px] items-center gap-4">
        {/* Agent info column */}
        <div className="flex items-center gap-4">
          {imageSrc?.[0] ? (
            <div className="relative aspect-video w-32 shrink-0 overflow-hidden rounded-[10px] bg-zinc-100">
              <Image
                src={imageSrc?.[0] ?? ""}
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
            <Text
              variant="h3"
              className="line-clamp-1 text-ellipsis text-neutral-800 dark:text-neutral-200"
              size="large-medium"
            >
              {agentName}
            </Text>
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
          {dateSubmitted}
        </div>

        {/* Status column */}
        <div data-testid="agent-status">
          <Status status={status} />
        </div>

        {/* Runs column */}
        <div className="text-right text-sm text-neutral-600 dark:text-neutral-400">
          {runs?.toLocaleString() ?? "0"}
        </div>

        {/* Reviews column */}
        <div className="text-right">
          {rating ? (
            <div className="flex items-center justify-end gap-1">
              <span className="text-sm font-medium">{rating.toFixed(1)}</span>
              <Star weight="fill" className="h-2 w-2" />
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
              <DropdownMenu.Item
                onSelect={handleView}
                className="flex cursor-pointer items-center rounded-md px-3 py-2 hover:bg-gray-100 dark:hover:bg-gray-700"
              >
                <Eye className="mr-2 h-4 w-4 dark:text-gray-100" />
                <span className="dark:text-gray-100">View</span>
              </DropdownMenu.Item>
              <DropdownMenu.Separator className="my-1 h-px bg-gray-300 dark:bg-gray-600" />
              <DropdownMenu.Item
                onSelect={handleDelete}
                className="flex cursor-pointer items-center rounded-md px-3 py-2 text-red-500 hover:bg-gray-100 dark:hover:bg-gray-700"
              >
                <Trash className="mr-2 h-4 w-4 text-red-500 dark:text-red-400" />
                <span className="dark:text-red-400">Delete</span>
              </DropdownMenu.Item>
            </DropdownMenu.Content>
          </DropdownMenu.Root>
        </div>
      </div>
    </div>
  );
};
