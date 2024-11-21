// AgentCard.tsx
import * as React from "react";
import Image from "next/image";
import { IconStarFilled, IconMore } from "@/components/ui/icons";
import { Status, StatusType } from "./Status";

export interface AgentTableCardProps {
  agentName: string;
  description: string;
  imageSrc: string;
  dateSubmitted: string;
  status: StatusType;
  runs?: number;
  rating?: number;
  onEdit: () => void;
}

export const AgentTableCard: React.FC<AgentTableCardProps> = ({
  agentName,
  description,
  imageSrc,
  dateSubmitted,
  status,
  runs,
  rating,
  onEdit,
}) => {
  return (
    <div className="border-b border-neutral-300 p-4">
      <div className="flex gap-4">
        <div className="relative h-[56px] w-[100px] overflow-hidden rounded-lg bg-[#d9d9d9]">
          <Image
            src={imageSrc}
            alt={agentName}
            layout="fill"
            objectFit="cover"
          />
        </div>
        <div className="flex-1">
          <h3 className="text-[15px] font-medium text-neutral-800">
            {agentName}
          </h3>
          <p className="line-clamp-2 text-sm text-neutral-600">{description}</p>
        </div>
        <button
          onClick={onEdit}
          className="h-fit rounded-full p-1 hover:bg-neutral-100"
        >
          <IconMore className="h-5 w-5 text-neutral-800" />
        </button>
      </div>

      <div className="mt-4 flex flex-wrap gap-4">
        <Status status={status} />
        <div className="text-sm text-neutral-600">{dateSubmitted}</div>
        {runs && (
          <div className="text-sm text-neutral-600">
            {runs.toLocaleString()} runs
          </div>
        )}
        {rating && (
          <div className="flex items-center gap-1">
            <span className="text-sm font-medium text-neutral-800">
              {rating.toFixed(1)}
            </span>
            <IconStarFilled className="h-4 w-4 text-neutral-800" />
          </div>
        )}
      </div>
    </div>
  );
};
