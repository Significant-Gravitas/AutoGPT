"use client";

import * as React from "react";
import Image from "next/image";
import { IconStarFilled, IconMore, IconEdit } from "@/components/ui/icons";
import { Status, StatusType } from "./Status";
import { TrashIcon } from "@radix-ui/react-icons";
import { StoreSubmissionRequest } from "@/lib/autogpt-server-api/types";
import { TableCell, TableRow } from "../ui/table";
import { Checkbox } from "../ui/checkbox";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "../ui/dropdown-menu";

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
  selectedAgents: Set<string>;
  setSelectedAgents: React.Dispatch<React.SetStateAction<Set<string>>>;
  onEditSubmission: (submission: StoreSubmissionRequest) => void;
  onDeleteSubmission: (submission_id: string) => void;
}

export const AgentTableRow: React.FC<AgentTableRowProps> = ({
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
  selectedAgents,
  setSelectedAgents,
  onEditSubmission,
  onDeleteSubmission,
}) => {
  // Create a unique ID for the checkbox
  const checkboxId = `agent-${id}-checkbox`;

  const handleEdit = React.useCallback(() => {
    onEditSubmission({
      agent_id,
      agent_version,
      slug: "",
      name: agentName,
      sub_heading,
      description,
      image_urls: imageSrc,
      categories: [],
    } satisfies StoreSubmissionRequest);
  }, [
    agent_id,
    agent_version,
    agentName,
    sub_heading,
    description,
    imageSrc,
    onEditSubmission,
  ]);

  const handleDelete = React.useCallback(() => {
    onDeleteSubmission(agent_id);
  }, [agent_id, onDeleteSubmission]);

  const handleCheckboxChange = React.useCallback(() => {
    if (selectedAgents.has(agent_id)) {
      selectedAgents.delete(agent_id);
    } else {
      selectedAgents.add(agent_id);
    }
    setSelectedAgents(new Set(selectedAgents));
  }, [agent_id, selectedAgents, setSelectedAgents]);

  return (
    <TableRow className="space-x-2.5 hover:bg-neutral-50 dark:hover:bg-neutral-800">
      <TableCell className="w-[40px]">
        <Checkbox
          id={checkboxId}
          aria-label={`Select ${agentName}`}
          checked={selectedAgents.has(agent_id)}
          onCheckedChange={handleCheckboxChange}
        />
        <label htmlFor={checkboxId} className="sr-only">
          Select {agentName}
        </label>
      </TableCell>

      <TableCell className="max-w-md">
        <div className="flex items-center gap-4">
          <div className="relative aspect-video w-[125px] min-w-[125px] overflow-hidden rounded-xl bg-[#d9d9d9] dark:bg-neutral-700">
            <Image
              src={imageSrc?.[0] ?? "/nada.png"}
              alt={agentName}
              fill
              style={{ objectFit: "cover" }}
            />
          </div>
          <div className="flex flex-col">
            <h3 className="line-clamp-2 font-sans text-sm font-medium text-neutral-800">
              {agentName}
            </h3>
            <p className="line-clamp-2 font-sans text-sm font-normal text-neutral-600">
              {description}
            </p>
          </div>
        </div>
      </TableCell>

      <TableCell className="font-sans text-sm font-normal text-neutral-600">
        {dateSubmitted}
      </TableCell>

      <TableCell>
        <Status status={status} />
      </TableCell>

      <TableCell className="text-right font-sans text-sm font-normal text-neutral-600">
        {runs?.toLocaleString() ?? "-"}
      </TableCell>

      <TableCell className="text-right font-sans text-sm font-normal text-neutral-600">
        {rating ? (
          <div className="flex items-center justify-end gap-1">
            <span className="text-sm font-medium text-neutral-800 dark:text-neutral-200">
              {rating.toFixed(1)}
            </span>
            <IconStarFilled className="h-4 w-4 text-neutral-800 dark:text-neutral-200" />
          </div>
        ) : (
          <span className="text-sm text-neutral-600 dark:text-neutral-400">
            No reviews
          </span>
        )}
      </TableCell>

      <TableCell className="text-right font-sans text-sm font-normal text-neutral-600">
        <DropdownMenu>
          <DropdownMenuTrigger data-testid="dropdown-button">
            <IconMore className="h-5 w-5 text-neutral-800 dark:text-neutral-200" />
          </DropdownMenuTrigger>
          <DropdownMenuContent className="z-10">
            <DropdownMenuItem onClick={handleEdit}>
              <div className="flex items-center font-sans text-sm font-normal text-neutral-600">
                <IconEdit className="mr-2 h-5 w-5" />
                <span>Edit</span>
              </div>
            </DropdownMenuItem>
            <DropdownMenuSeparator />
            <DropdownMenuItem
              onClick={handleDelete}
              className="font-sans text-sm font-normal text-red-500"
            >
              <TrashIcon className="mr-2 h-5 w-5 text-red-500" />
              <span>Delete</span>
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </TableCell>
    </TableRow>
  );
};
