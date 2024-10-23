// AgentCard.tsx
import * as React from "react";
import Image from "next/image";
import { Button } from "./Button";
import { IconStarFilled, IconEdit } from "@/components/ui/icons";
import { Card } from "@/components/ui/card";
import { AgentTableRowProps } from "./AgentTableRow";

export const AgentTableCard: React.FC<AgentTableRowProps> = ({
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
    <Card className="mb-4 p-4">
      <div className="flex">
        <div className="relative mr-4 h-20 w-20 overflow-hidden rounded-xl">
          <Image
            src={imageSrc}
            alt={agentName}
            layout="fill"
            objectFit="cover"
          />
        </div>
        <div className="flex-1">
          <h3 className="mb-2 font-neue text-lg font-medium tracking-tight text-[#272727]">
            {agentName}
          </h3>
          <p className="mb-2 font-neue text-sm leading-tight tracking-tight text-[#282828]">
            {description}
          </p>
          <div className="mb-2 font-neue text-sm leading-tight tracking-tight text-[#282828]">
            <strong>Date submitted:</strong> {dateSubmitted}
          </div>
          <div className="mb-2 font-neue text-sm leading-tight tracking-tight text-[#282828]">
            <strong>Status:</strong> {status}
          </div>
          {runs !== undefined && (
            <div className="mb-2 font-neue text-sm leading-tight tracking-tight text-[#282828]">
              <strong>Runs:</strong> {runs.toLocaleString()}
            </div>
          )}
          {rating !== undefined && (
            <div className="mb-2 flex items-center font-neue text-sm leading-tight tracking-tight text-[#282828]">
              <strong>Rating:</strong>
              <span className="ml-2">{rating.toFixed(1)}</span>
              <IconStarFilled className="ml-1" />
            </div>
          )}
          <Button
            variant="outline"
            size="sm"
            className="mt-2 flex items-center gap-1"
            onClick={onEdit}
          >
            <IconEdit />
            <span>Edit</span>
          </Button>
        </div>
      </div>
    </Card>
  );
};
