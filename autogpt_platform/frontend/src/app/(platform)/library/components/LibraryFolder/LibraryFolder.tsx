"use client";

import { Text } from "@/components/atoms/Text/Text";
import { Button } from "@/components/atoms/Button/Button";
import { FolderIcon, FolderColor } from "./FolderIcon";
import { useState } from "react";
import { PencilSimpleIcon, TrashIcon } from "@phosphor-icons/react";
import type { AgentStatus } from "../../types";
import { StatusBadge } from "../StatusBadge/StatusBadge";

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
  /** Worst status among child agents (optional, for status aggregation). */
  worstStatus?: AgentStatus;
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
  worstStatus,
}: Props) {
  const [isHovered, setIsHovered] = useState(false);
  const [isDragOver, setIsDragOver] = useState(false);

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
      className={`group relative inline-flex h-[10.625rem] w-full max-w-[25rem] cursor-pointer flex-col items-start justify-between gap-2.5 rounded-medium border p-4 shadow-sm backdrop-blur-md transition-all duration-200 hover:shadow-md ${
        isDragOver
          ? "border-blue-400 bg-blue-50 ring-2 ring-blue-200"
          : "border-indigo-200/40 bg-gradient-to-br from-indigo-50/40 via-white/70 to-purple-50/30"
      }`}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      onClick={onClick}
    >
      <div className="flex w-full items-center justify-between gap-4">
        {/* Left side - Folder name and agent count */}
        <div className="flex flex-1 flex-col gap-2">
          <Text
            variant="h5"
            data-testid="library-folder-name"
            className="line-clamp-2 hyphens-auto break-words"
          >
            {name}
          </Text>
          <div className="flex items-center gap-2">
            <Text
              variant="small"
              className="text-zinc-500"
              data-testid="library-folder-agent-count"
            >
              {agentCount} {agentCount === 1 ? "agent" : "agents"}
            </Text>
            {worstStatus && worstStatus !== "idle" && (
              <StatusBadge status={worstStatus} />
            )}
          </div>
        </div>

        {/* Right side - Custom folder icon */}
        <div className="relative top-5 flex flex-shrink-0 items-center">
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
          className="h-8 w-8 border border-neutral-200 bg-white/80 p-2 text-neutral-500 hover:bg-white hover:text-neutral-700"
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
          className="h-8 w-8 border border-neutral-200 bg-white/80 p-2 text-neutral-500 hover:bg-white hover:text-neutral-700"
        >
          <TrashIcon className="h-4 w-4" />
        </Button>
      </div>
    </div>
  );
}
