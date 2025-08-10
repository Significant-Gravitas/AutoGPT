"use client";

import Image from "next/image";
import { IconStarFilled, IconMore } from "@/components/ui/icons";
import { StoreSubmission } from "@/app/api/__generated__/models/storeSubmission";
import { Status, StatusType } from "@/components/agptui/Status";
import { SubmissionStatus } from "@/app/api/__generated__/models/submissionStatus";

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
  onViewSubmission: (submission: StoreSubmission) => void;
}

export const AgentTableCard = ({
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
  onViewSubmission,
}: AgentTableCardProps) => {
  const onView = () => {
    onViewSubmission({
      agent_id,
      agent_version,
      slug: "",
      name: agentName,
      sub_heading,
      description,
      image_urls: imageSrc,
      date_submitted: dateSubmitted,
      // SafeCast: status is a string from the API...
      status: status.toUpperCase() as SubmissionStatus,
      runs,
      rating,
    });
  };

  return (
    <div className="border-b border-neutral-300 p-4 dark:border-neutral-700">
      <div className="flex gap-4">
        <div className="relative aspect-video w-24 shrink-0 overflow-hidden rounded-lg bg-[#d9d9d9] dark:bg-neutral-800">
          <Image
            src={imageSrc?.[0] ?? "/nada.png"}
            alt={agentName}
            fill
            style={{ objectFit: "cover" }}
          />
        </div>
        <div className="flex-1">
          <h3 className="text-[15px] font-medium text-neutral-800 dark:text-neutral-200">
            {agentName}
          </h3>
          <p className="line-clamp-2 text-sm text-neutral-600 dark:text-neutral-400">
            {description}
          </p>
        </div>
        <button
          onClick={onView}
          className="h-fit rounded-full p-1 hover:bg-neutral-100 dark:hover:bg-neutral-700"
        >
          <IconMore className="h-5 w-5 text-neutral-800 dark:text-neutral-200" />
        </button>
      </div>

      <div className="mt-4 flex flex-wrap gap-4">
        <Status status={status} />
        <div className="text-sm text-neutral-600 dark:text-neutral-400">
          {dateSubmitted}
        </div>
        <div className="text-sm text-neutral-600 dark:text-neutral-400">
          {runs.toLocaleString()} runs
        </div>
        <div className="flex items-center gap-1">
          <span className="text-sm font-medium text-neutral-800 dark:text-neutral-200">
            {rating.toFixed(1)}
          </span>
          <IconStarFilled className="h-4 w-4 text-neutral-800 dark:text-neutral-200" />
        </div>
      </div>
    </div>
  );
};
