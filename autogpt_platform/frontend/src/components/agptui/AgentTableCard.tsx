"use client";

import * as React from "react";
import Image from "next/image";
import { IconStarFilled, IconMore } from "@/components/ui/icons";
import { Status, StatusType } from "./Status";
import { StoreSubmissionRequest } from "@/lib/autogpt-server-api";

export interface AgentTableCardProps {
  agent_id: string;
  agent_version: number;
  agentName: string;
  sub_heading: string;
  description: string;
  imageSrc: string[];
  dateSubmitted: string;
  status: StatusType;
  runs: number;
  rating: number;
  id: number;
  onEditSubmission: (submission: StoreSubmissionRequest) => void;
}

export const AgentTableCard: React.FC<AgentTableCardProps> = ({
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
  onEditSubmission,
}) => {
  const onEdit = () => {
    console.log("Edit agent", agentName);
    onEditSubmission({
      agent_id,
      agent_version,
      slug: "",
      name: agentName,
      sub_heading,
      description,
      image_urls: imageSrc,
      categories: [],
    });
  };

  return (
    <div
      className="border-b border-neutral-300 p-4 dark:border-neutral-700"
      data-testid="agent-table-card"
    >
      <div className="flex gap-4">
        <div className="relative h-[56px] w-[100px] overflow-hidden rounded-lg bg-[#d9d9d9] dark:bg-neutral-800">
          <Image
            src={imageSrc?.[0] ?? "/nada.png"}
            alt={agentName}
            fill
            style={{ objectFit: "cover" }}
          />
        </div>
        <div className="flex-1">
          <h3 className="font-sans text-sm font-medium text-neutral-600">
            {agentName}
          </h3>
          <p className="font-sans text-sm font-normal text-neutral-600">
            {description}
          </p>
        </div>
        <button
          onClick={onEdit}
          className="h-fit rounded-full p-1 hover:bg-neutral-100 dark:hover:bg-neutral-700"
        >
          <IconMore className="h-5 w-5 text-neutral-800 dark:text-neutral-200" />
        </button>
      </div>

      <div className="mt-4 flex flex-wrap items-center gap-4">
        <Status status={status} />
        <div className="font-sans text-sm font-normal text-neutral-600">
          {dateSubmitted}
        </div>
        <div className="font-sans text-sm font-normal text-neutral-600">
          {runs.toLocaleString()} runs
        </div>
        <div className="flex items-center gap-1">
          <span className="font-sans text-sm font-normal text-neutral-600">
            {rating.toFixed(1)}
          </span>
          <IconStarFilled className="h-4 w-4 text-neutral-800 dark:text-neutral-200" />
        </div>
      </div>
    </div>
  );
};
