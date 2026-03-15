"use client";

import { Text } from "@/components/atoms/Text/Text";
import { Button } from "@/components/atoms/Button/Button";
import {
  FolderIcon,
  FolderColor,
  folderCardStyles,
  resolveColor,
} from "./FolderIcon";
import { useState } from "react";
import { PencilSimpleIcon, TrashIcon } from "@phosphor-icons/react";

interface Props {
  id: string;
  name: string;
  agentCount: number;
  color?: FolderColor;
  icon: string;
  onEdit?: () => void;
  onDelete?: () => void;
  onAgentDrop?: (agentId: string, folderId: string) => void;
  onClick?: () => void;
}

export function LibraryFolder({
  id,
  name,
  agentCount,
  color,
  icon,
  onEdit,
  onDelete,
  onAgentDrop,
  onClick,
}: Props) {
  const [isHovered, setIsHovered] = useState(false);
  const [isDragOver, setIsDragOver] = useState(false);
  const resolvedColor = resolveColor(color);
  const cardStyle = folderCardStyles[resolvedColor];

  function handleDragOver(e: React.DragEvent<HTMLDivElement>) {
    if (e.dataTransfer.types.includes("application/agent-id")) {
      e.preventDefault();
      e.dataTransfer.dropEffect = "move";
      setIsDragOver(true);
    }
  }

  function handleDragLeave() {
    setIsDragOver(false);
  }

  function handleDrop(e: React.DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setIsDragOver(false);
    const agentId = e.dataTransfer.getData("application/agent-id");
    if (agentId && onAgentDrop) {
      onAgentDrop(agentId, id);
    }
  }

  return (
    <div
      data-testid="library-folder"
      data-folder-id={id}
      className={`group relative inline-flex h-[10.625rem] w-full max-w-[25rem] cursor-pointer flex-col items-start justify-between gap-2.5 rounded-medium border p-4 transition-all duration-200 hover:shadow-md ${
        isDragOver
          ? "border-blue-400 bg-blue-50 ring-2 ring-blue-200"
          : `${cardStyle.border} ${cardStyle.bg}`
      }`}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      onClick={onClick}
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
          aria-label="Edit folder"
          onClick={(e) => {
            e.stopPropagation();
            onEdit?.();
          }}
          className={`h-8 w-8 border p-2 ${cardStyle.buttonBase} ${cardStyle.buttonHover}`}
        >
          <PencilSimpleIcon className="h-4 w-4" />
        </Button>
        <Button
          variant="icon"
          size="icon"
          aria-label="Delete folder"
          onClick={(e) => {
            e.stopPropagation();
            onDelete?.();
          }}
          className={`h-8 w-8 border p-2 ${cardStyle.buttonBase} ${cardStyle.buttonHover}`}
        >
          <TrashIcon className="h-4 w-4" />
        </Button>
      </div>
    </div>
  );
}
