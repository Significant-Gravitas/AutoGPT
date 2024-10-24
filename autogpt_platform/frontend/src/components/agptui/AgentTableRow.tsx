import * as React from "react";
import Image from "next/image";
import { Button } from "./Button";
import { IconStarFilled, IconEdit } from "@/components/ui/icons";

export interface AgentTableRowProps {
  agentName: string;
  description: string;
  imageSrc: string;
  dateSubmitted: string;
  status: string;
  runs?: number;
  rating?: number;
  onEdit: () => void;
}

export const AgentTableRow: React.FC<AgentTableRowProps> = ({
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
    <tr className="border-b border-[#d9d9d9] py-4">
      <td className="flex items-center">
        <div className="relative my-4 mr-4 h-20 w-20 overflow-hidden rounded-xl sm:h-20 sm:w-[125px]">
          <Image
            src={imageSrc}
            alt={agentName}
            layout="fill"
            objectFit="cover"
          />
        </div>
        <div className="max-w-[293px]">
          <h3 className="mb-2 font-neue text-lg font-medium tracking-tight text-[#272727]">
            {agentName}
          </h3>
          <p className="font-neue text-sm leading-tight tracking-tight text-[#282828]">
            {description}
          </p>
        </div>
      </td>
      <td className="font-neue text-base leading-[21px] tracking-tight text-[#282828]">
        {dateSubmitted}
      </td>
      <td className="font-neue text-base leading-[21px] tracking-tight text-[#282828]">
        {status}
      </td>
      <td className="font-neue text-base leading-[21px] tracking-tight text-[#282828]">
        {runs !== undefined ? runs.toLocaleString() : ""}
      </td>
      <td>
        {rating !== undefined && (
          <div className="flex items-center">
            <span className="mr-2 font-neue text-base font-medium tracking-tight text-[#272727]">
              {rating.toFixed(1)}
            </span>
            <IconStarFilled />
          </div>
        )}
      </td>
      <td>
        <Button
          variant="outline"
          size="sm"
          className="flex items-center gap-1"
          onClick={onEdit}
        >
          <IconEdit />
          <span className="font-neue text-sm">Edit</span>
        </Button>
      </td>
    </tr>
  );
};
