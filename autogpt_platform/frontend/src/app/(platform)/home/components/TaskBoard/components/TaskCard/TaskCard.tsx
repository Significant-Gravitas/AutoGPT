import type { ExpertIdentity } from "@/app/(platform)/copilot/useExpertMap";
import type { SessionSummaryResponse } from "@/app/api/__generated__/models/sessionSummaryResponse";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuTrigger,
} from "@/components/molecules/DropdownMenu/DropdownMenu";
import { formatTimeAgo } from "@/lib/utils/time";
import {
  Clock01Icon,
  FileAttachmentIcon,
  MoreHorizontalIcon,
} from "@hugeicons/core-free-icons";
import Link from "next/link";
import type { DragEvent } from "react";
import {
  type BoardColumnDef,
  formatRunMoment,
  getPreviewLine,
  isSessionRunning,
  TASK_CARD_MIME,
} from "../../helpers";
import { ExpertByline } from "../ExpertByline/ExpertByline";
import { StatusIcon } from "../StatusIcon/StatusIcon";

interface Props {
  session: SessionSummaryResponse;
  columnId: string;
  allColumns: BoardColumnDef[];
  expert: ExpertIdentity | undefined;
  nextFollowup: string | undefined;
  onMove: (columnId: string) => void;
}

export function TaskCard({
  session,
  columnId,
  allColumns,
  expert,
  nextFollowup,
  onMove,
}: Props) {
  const title = session.title || "Untitled chat";
  const isRunning = isSessionRunning(session);
  const fileCount = session.file_count ?? 0;

  function handleDragStart(e: DragEvent<HTMLElement>) {
    e.dataTransfer.setData(TASK_CARD_MIME, session.id);
    e.dataTransfer.effectAllowed = "move";
  }

  return (
    <article
      draggable
      onDragStart={handleDragStart}
      className="group relative cursor-grab rounded-lg border border-zinc-200/80 bg-white shadow-[0_1px_2px_rgba(0,0,0,0.04)] transition-colors hover:border-zinc-300 active:cursor-grabbing"
    >
      <Link
        href={`/home?sessionId=${session.id}`}
        draggable={false}
        className="block rounded-lg px-3.5 pb-3 pt-3 outline-none focus-visible:ring-2 focus-visible:ring-violet-400"
      >
        <ExpertByline expert={expert} />
        <div className="mt-2.5 flex items-start gap-2">
          <StatusIcon columnId={columnId} className="mt-[0.2rem]" />
          <Text variant="body-medium" tone="primary" className="line-clamp-2">
            {title}
          </Text>
        </div>
        <Text variant="small" tone="secondary" className="mt-1 truncate">
          {session.last_message
            ? getPreviewLine(session.last_message)
            : "Not started yet"}
        </Text>
        {isRunning && (
          <div className="mt-2.5 flex">
            <span className="inline-flex items-center gap-1.5 rounded-full border border-zinc-200 px-2 py-0.5">
              <span
                aria-hidden
                className="size-1.5 animate-pulse rounded-full bg-amber-500"
              />
              <Text variant="small" tone="secondary">
                Working
              </Text>
            </span>
          </div>
        )}
        <div className="mt-2.5 flex items-center gap-3">
          {nextFollowup && (
            <span
              className="inline-flex items-center gap-1 text-zinc-500"
              aria-label={`Follow-up scheduled for ${formatRunMoment(nextFollowup)}`}
            >
              <Icon icon={Clock01Icon} size={14} />
              <Text variant="small" tone="muted" aria-hidden>
                {formatRunMoment(nextFollowup)}
              </Text>
            </span>
          )}
          {fileCount > 0 && (
            <span
              className="inline-flex items-center gap-1 text-zinc-500"
              aria-label={`${fileCount} ${fileCount === 1 ? "file" : "files"}`}
            >
              <Icon icon={FileAttachmentIcon} size={14} />
              <Text
                variant="small"
                tone="muted"
                aria-hidden
                className="tabular-nums"
              >
                {fileCount}
              </Text>
            </span>
          )}
          <Text variant="small" tone="muted" className="ml-auto">
            {formatTimeAgo(session.updated_at)}
          </Text>
        </div>
      </Link>
      <div className="absolute right-1.5 top-1.5 opacity-0 transition-opacity focus-within:opacity-100 group-hover:opacity-100 max-md:opacity-100">
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button
              variant="ghost"
              size="icon-xs"
              className="text-zinc-500"
              aria-label={`Move ${title}`}
            >
              <Icon icon={MoreHorizontalIcon} size={16} />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end">
            <DropdownMenuLabel>Move to</DropdownMenuLabel>
            {allColumns
              .filter((column) => column.id !== columnId)
              .map((column) => (
                <DropdownMenuItem
                  key={column.id}
                  onSelect={() => onMove(column.id)}
                  className="gap-2"
                >
                  <StatusIcon columnId={column.id} />
                  {column.name}
                </DropdownMenuItem>
              ))}
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
    </article>
  );
}
