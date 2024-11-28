"use client";

import * as React from "react";
import Image from "next/image";
import { Button } from "./Button";
import { IconStarFilled, IconEdit, IconMore } from "@/components/ui/icons";
import { Status, StatusType } from "./Status";

export interface AgentTableRowProps {
  agentName: string;
  description: string;
  imageSrc: string;
  dateSubmitted: string;
  status: StatusType;
  runs?: number;
  rating?: number;
  id: number;
}

export const AgentTableRow: React.FC<AgentTableRowProps> = ({
  agentName,
  description,
  imageSrc,
  dateSubmitted,
  status,
  runs,
  rating,
  id,
}) => {
  // Create a unique ID for the checkbox
  const checkboxId = `agent-${id}-checkbox`;

  const onEdit = () => {
    console.log("Edit agent", id);
  };

  return (
    <div className="hidden items-center border-b border-neutral-300 px-4 py-4 hover:bg-neutral-50 dark:border-neutral-700 dark:hover:bg-neutral-800 md:flex">
      <div className="flex items-center">
        <div className="flex items-center">
          <input
            type="checkbox"
            id={checkboxId}
            aria-label={`Select ${agentName}`}
            className="mr-4 h-5 w-5 rounded border-2 border-neutral-400 dark:border-neutral-600"
          />
          {/* Single label instead of multiple */}
          <label htmlFor={checkboxId} className="sr-only">
            Select {agentName}
          </label>
        </div>
      </div>

      <div className="grid w-full grid-cols-[minmax(400px,1fr),180px,140px,100px,100px,40px] items-center gap-4">
        {/* Agent info column */}
        <div className="flex items-center gap-4">
          <div className="relative h-[70px] w-[125px] overflow-hidden rounded-[10px] bg-[#d9d9d9] dark:bg-neutral-700">
            <Image
              src={imageSrc}
              alt={agentName}
              fill
              style={{ objectFit: "cover" }}
            />
          </div>
          <div className="flex flex-col">
            <h3 className="text-[15px] font-medium text-neutral-800 dark:text-neutral-200">
              {agentName}
            </h3>
            <p className="line-clamp-2 text-sm text-neutral-600 dark:text-neutral-400">
              {description}
            </p>
          </div>
        </div>

        {/* Date column */}
        <div className="pl-14 text-sm text-neutral-600 dark:text-neutral-400">{dateSubmitted}</div>

        {/* Status column */}
        <div>
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
              <span className="text-sm font-medium text-neutral-800 dark:text-neutral-200">
                {rating.toFixed(1)}
              </span>
              <IconStarFilled className="h-4 w-4 text-neutral-800 dark:text-neutral-200" />
            </div>
          ) : (
            <span className="text-sm text-neutral-600 dark:text-neutral-400">No reviews</span>
          )}
        </div>

        {/* Actions - Three dots menu */}
        <div className="flex justify-end">
          <button
            onClick={onEdit}
            className="rounded-full p-1 hover:bg-neutral-100 dark:hover:bg-neutral-700"
          >
            <IconMore className="h-5 w-5 text-neutral-800 dark:text-neutral-200" />
          </button>
        </div>
      </div>
    </div>
  );
};
