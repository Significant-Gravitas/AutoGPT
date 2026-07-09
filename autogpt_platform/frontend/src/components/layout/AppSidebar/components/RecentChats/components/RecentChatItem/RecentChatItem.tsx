"use client";

import { ChatOriginIcon } from "@/app/(platform)/copilot/components/ChatOriginIcon/ChatOriginIcon";
import { resolvePlatformLogo } from "@/app/(platform)/copilot/components/ChatOriginIcon/platformLogos";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/molecules/DropdownMenu/DropdownMenu";
import {
  SidebarMenuAction,
  SidebarMenuButton,
  SidebarMenuItem,
} from "@/components/ui/sidebar";
import {
  CircleNotchIcon,
  DotsThreeIcon,
  DownloadSimpleIcon,
  PencilSimpleIcon,
  ShareNetworkIcon,
  TrashIcon,
} from "@phosphor-icons/react";
import Link from "next/link";
import { useRef } from "react";

interface Session {
  id: string;
  title?: string | null;
  source_platform?: string | null;
  is_processing?: boolean | null;
  updated_at: string;
}

interface Props {
  session: Session;
  isActive: boolean;
  isEditing: boolean;
  editingTitle: string;
  onEditingTitleChange: (value: string) => void;
  onSubmitRename: (id: string) => void;
  onCancelRename: () => void;
  isExporting: boolean;
  isDeleting: boolean;
  chatSharingEnabled: boolean;
  onRename: (id: string, title: string | null | undefined) => void;
  onExport: (id: string, title: string | null | undefined) => void;
  onShare: (id: string) => void;
  onDelete: (id: string, title: string | null | undefined) => void;
}

export function RecentChatItem({
  session,
  isActive,
  isEditing,
  editingTitle,
  onEditingTitleChange,
  onSubmitRename,
  onCancelRename,
  isExporting,
  isDeleting,
  chatSharingEnabled,
  onRename,
  onExport,
  onShare,
  onDelete,
}: Props) {
  const title = session.title || "Untitled chat";
  const hasPlatformLogo = !!resolvePlatformLogo(session.source_platform);

  // Enter and Escape both blur the input, which would otherwise trigger a
  // second (or contradictory) submit via onBlur. This flag lets onBlur skip
  // the submit when the key handler already resolved the edit.
  const skipBlurSubmitRef = useRef(false);

  if (isEditing) {
    return (
      <SidebarMenuItem>
        <input
          autoFocus
          aria-label="Rename chat"
          value={editingTitle}
          onChange={(e) => onEditingTitleChange(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              e.preventDefault();
              skipBlurSubmitRef.current = true;
              onSubmitRename(session.id);
              e.currentTarget.blur();
            } else if (e.key === "Escape") {
              e.preventDefault();
              skipBlurSubmitRef.current = true;
              onCancelRename();
              e.currentTarget.blur();
            }
          }}
          onBlur={() => {
            if (skipBlurSubmitRef.current) {
              skipBlurSubmitRef.current = false;
              return;
            }
            onSubmitRename(session.id);
          }}
          className="w-full rounded-md border border-zinc-300 bg-white px-2 py-1 text-sm text-zinc-800 outline-none focus:border-purple-500 focus:ring-1 focus:ring-purple-500"
        />
      </SidebarMenuItem>
    );
  }

  return (
    <SidebarMenuItem>
      <SidebarMenuButton
        asChild
        isActive={isActive}
        tooltip={title}
        className="font-normal data-[active=true]:!bg-zinc-100 data-[active=true]:font-normal hover:!bg-zinc-100"
      >
        <Link href={`/copilot?sessionId=${session.id}`}>
          {session.is_processing ? (
            <LoadingSpinner
              size="small"
              className="size-4 shrink-0 text-purple-600"
            />
          ) : hasPlatformLogo ? (
            <ChatOriginIcon sourcePlatform={session.source_platform} />
          ) : null}
          <span className="truncate">{title}</span>
        </Link>
      </SidebarMenuButton>

      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <SidebarMenuAction
            showOnHover
            aria-label="Chat actions"
            className="border border-zinc-200 bg-white"
          >
            <DotsThreeIcon weight="bold" />
          </SidebarMenuAction>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end">
          <DropdownMenuItem onClick={() => onRename(session.id, session.title)}>
            <PencilSimpleIcon className="mr-2 h-4 w-4" />
            Rename
          </DropdownMenuItem>
          <DropdownMenuItem
            onClick={() => onExport(session.id, session.title)}
            onSelect={(e) => {
              if (isExporting) e.preventDefault();
            }}
            disabled={isExporting}
          >
            {isExporting ? (
              <CircleNotchIcon className="mr-2 h-4 w-4 animate-spin" />
            ) : (
              <DownloadSimpleIcon className="mr-2 h-4 w-4" />
            )}
            {isExporting ? "Exporting…" : "Export chat"}
          </DropdownMenuItem>
          {chatSharingEnabled && (
            <DropdownMenuItem onClick={() => onShare(session.id)}>
              <ShareNetworkIcon className="mr-2 h-4 w-4" />
              Share chat
            </DropdownMenuItem>
          )}
          <DropdownMenuItem
            onClick={() => onDelete(session.id, session.title)}
            disabled={isDeleting}
            className="text-red-600 focus:bg-red-50 focus:text-red-600"
          >
            <TrashIcon className="mr-2 h-4 w-4" />
            Delete chat
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    </SidebarMenuItem>
  );
}
