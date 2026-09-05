"use client";

import type { SessionSummaryResponse } from "@/app/api/__generated__/models/sessionSummaryResponse";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/molecules/DropdownMenu/DropdownMenu";
import { cn } from "@/lib/utils";
import { AnimatePresence, motion } from "framer-motion";
import type { MouseEvent, RefObject } from "react";
import { ChatSessionBlock } from "../../../ChatSessionBlock/ChatSessionBlock";
import {
  Delete02Icon,
  Download04Icon,
  Loading03Icon,
  MoreHorizontalIcon,
  PencilIcon,
  PinIcon,
  PinOffIcon,
  Share03Icon,
} from "@hugeicons/core-free-icons";
import { Icon } from "@/components/atoms/Icon/Icon";

interface Props {
  session: SessionSummaryResponse;
  isActive: boolean;
  /** The row below is the active one, which owns the divider instead. */
  isNextActive: boolean;
  isEditing: boolean;
  editingTitle: string;
  renameInputRef: RefObject<HTMLInputElement>;
  isExporting: boolean;
  isDeleting: boolean;
  isPinningEnabled: boolean;
  isSharingEnabled: boolean;
  showProcessing: boolean;
  showCompleted: boolean;
  onSelect: () => void;
  onEditingTitleChange: (title: string) => void;
  onRenameCancel: () => void;
  onRenameBlur: () => void;
  onPin: (e: MouseEvent) => void;
  onRename: (e: MouseEvent) => void;
  onExport: (e: MouseEvent) => void;
  onShare: (e: MouseEvent) => void;
  onDelete: (e: MouseEvent) => void;
}

export function ChatSessionRow({
  session,
  isActive,
  isNextActive,
  isEditing,
  editingTitle,
  renameInputRef,
  isExporting,
  isDeleting,
  isPinningEnabled,
  isSharingEnabled,
  showProcessing,
  showCompleted,
  onSelect,
  onEditingTitleChange,
  onRenameCancel,
  onRenameBlur,
  onPin,
  onRename,
  onExport,
  onShare,
  onDelete,
}: Props) {
  return (
    <div
      className={cn(
        "group relative w-full transition-colors",
        isActive
          ? "rounded-lg bg-zinc-100"
          : cn(
              "border-b border-b-[#8080800f] last:border-b-0 hover:bg-zinc-50",
              isNextActive && "!border-b-0",
            ),
      )}
    >
      {isEditing ? (
        <div className="px-3 py-2.5">
          <input
            ref={renameInputRef}
            type="text"
            aria-label="Rename chat"
            value={editingTitle}
            onChange={(e) => onEditingTitleChange(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                e.currentTarget.blur();
              } else if (e.key === "Escape") {
                onRenameCancel();
              }
            }}
            onBlur={onRenameBlur}
            className="w-full rounded border border-zinc-300 bg-white px-2 py-1 text-sm text-zinc-800 outline-none focus:border-purple-500 focus:ring-1 focus:ring-purple-500"
          />
        </div>
      ) : (
        <button
          onClick={onSelect}
          className={cn(
            "w-full px-3 py-2.5 text-left",
            isExporting ? "pr-[68px]" : "pr-10",
          )}
        >
          <ChatSessionBlock
            title={session.title}
            titleContent={
              <AnimatePresence mode="wait" initial={false}>
                <motion.span
                  key={session.title || "untitled"}
                  initial={{ opacity: 0, y: 4 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -4 }}
                  transition={{ duration: 0.2 }}
                  className="block truncate"
                >
                  {session.title || "Untitled chat"}
                </motion.span>
              </AnimatePresence>
            }
            updatedAt={session.updated_at}
            sourcePlatform={session.source_platform}
            showPinned={isPinningEnabled && !!session.is_pinned}
            isActive={isActive}
            chatStatus={session.chat_status}
            showProcessing={showProcessing}
            showCompleted={showCompleted}
          />
        </button>
      )}
      {isExporting && (
        <div
          className="pointer-events-none absolute right-9 top-1/2 z-10 -translate-y-1/2 rounded-full bg-white text-zinc-600 shadow-sm"
          aria-label="Exporting chat"
          title="Exporting chat…"
        >
          <div className="relative h-7 w-7">
            <div className="absolute inset-0 animate-spin rounded-full border-2 border-zinc-200 border-t-zinc-700" />
            <Icon
              icon={Download04Icon}
              className="absolute inset-0 m-auto h-3.5 w-3.5"
            />
          </div>
        </div>
      )}
      {!isEditing && (
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <button
              onClick={(e) => e.stopPropagation()}
              className="absolute right-2 top-1/2 -translate-y-1/2 rounded-full p-1.5 text-zinc-600 transition-all hover:bg-neutral-100"
              aria-label="More actions"
            >
              <Icon icon={MoreHorizontalIcon} className="h-4 w-4" />
            </button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            {isPinningEnabled && (
              <DropdownMenuItem onClick={onPin}>
                {session.is_pinned ? (
                  <>
                    <Icon icon={PinOffIcon} className="mr-2 h-4 w-4" />
                    Unpin chat
                  </>
                ) : (
                  <>
                    <Icon icon={PinIcon} className="mr-2 h-4 w-4" />
                    Pin chat
                  </>
                )}
              </DropdownMenuItem>
            )}
            <DropdownMenuItem onClick={onRename}>
              <Icon icon={PencilIcon} className="mr-2 h-4 w-4" />
              Rename
            </DropdownMenuItem>
            <DropdownMenuItem
              onClick={onExport}
              onSelect={(e) => {
                if (isExporting) e.preventDefault();
              }}
              disabled={isExporting}
            >
              {isExporting ? (
                <Icon
                  icon={Loading03Icon}
                  className="mr-2 h-4 w-4 animate-spin"
                />
              ) : (
                <Icon icon={Download04Icon} className="mr-2 h-4 w-4" />
              )}
              {isExporting ? "Exporting…" : "Export chat"}
            </DropdownMenuItem>
            {isSharingEnabled && (
              <DropdownMenuItem onClick={onShare}>
                <Icon icon={Share03Icon} className="mr-2 h-4 w-4" />
                Share chat
              </DropdownMenuItem>
            )}
            <DropdownMenuItem
              onClick={onDelete}
              disabled={isDeleting}
              className="text-red-600 focus:bg-red-50 focus:text-red-600"
            >
              <Icon icon={Delete02Icon} className="mr-2 h-4 w-4" />
              Delete chat
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      )}
    </div>
  );
}
