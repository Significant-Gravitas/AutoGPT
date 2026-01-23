"use client";

import { Text } from "@/components/atoms/Text/Text";
import { Button } from "@/components/atoms/Button/Button";
import { FolderIcon, FolderColor } from "./FolderIcon";
import { useState } from "react";
import {
  PencilSimpleIcon,
  TrashIcon,
  StarIcon,
} from "@phosphor-icons/react";

interface Props {
  name: string;
  agentCount: number;
  color: FolderColor;
  icon: string;
  onEdit?: () => void;
  onDelete?: () => void;
  onFavorite?: () => void;
  isFavorite?: boolean;
}

export function LibraryFolder({
  name,
  agentCount,
  color,
  icon,
  onEdit,
  onDelete,
  onFavorite,
  isFavorite = false,
}: Props) {
  const [isHovered, setIsHovered] = useState(false);

  return (
    <div
      data-testid="library-folder"
      className="group relative inline-flex h-[10.625rem] w-full max-w-[25rem] cursor-pointer flex-col items-start justify-start gap-2.5 rounded-medium border border-zinc-100 bg-white p-4 transition-all duration-200 hover:shadow-md"
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      <div className="flex w-full items-start justify-between gap-4">
        {/* Left side - Folder name and agent count */}
        <div className="flex flex-1 flex-col gap-2">
          <Text
            variant="h5"
            data-testid="library-folder-name"
            className="line-clamp-2 hyphens-auto break-words"
          >
            {name}
          </Text>
          <Text
            variant="small"
            className="text-zinc-500"
            data-testid="library-folder-agent-count"
          >
            {agentCount} {agentCount === 1 ? "agent" : "agents"}
          </Text>
        </div>

        {/* Right side - Custom folder icon */}
        <div className="flex-shrink-0">
          <FolderIcon isOpen={isHovered} color={color} icon={icon} />
        </div>
      </div>

      {/* Action buttons - visible on hover */}
      <div
       className="flex items-center justify-end gap-2"
        data-testid="library-folder-actions"
      >
        <Button
          variant="icon"
          size="icon"
          aria-label="Favorite agent"
          onClick={(e) => {
            e.stopPropagation();
            onFavorite?.();
          }}
          className="h-8 w-8 p-2"
        >
          <StarIcon
            className="h-4 w-4"
            weight={isFavorite ? "fill" : "regular"}
            color={isFavorite ? "#facc15" : "currentColor"}
          />
        </Button>
        <Button
          variant="icon"
          size="icon"
          aria-label="Edit agent"
          onClick={(e) => {
            e.stopPropagation();
            onEdit?.();
          }}
          className="h-8 w-8 p-2"
        >
          <PencilSimpleIcon className="h-4 w-4" />
        </Button>
        <Button
          variant="icon"
          size="icon"
          aria-label="Delete agent"
          onClick={(e) => {
            e.stopPropagation();
            onDelete?.();
          }}
          className="h-8 w-8 p-2 hover:border-red-300 hover:bg-red-50 hover:text-red-600"
        >
          <TrashIcon className="h-4 w-4" />
        </Button>
      </div>
    </div>
  );
}
