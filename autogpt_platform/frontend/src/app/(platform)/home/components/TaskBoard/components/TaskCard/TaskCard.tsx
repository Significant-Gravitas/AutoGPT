import type { ExpertIdentity } from "@/app/(platform)/copilot/useExpertMap";
import type { SessionSummaryResponse } from "@/app/api/__generated__/models/sessionSummaryResponse";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { AutopilotAvatar } from "@/components/molecules/AutopilotAvatar/AutopilotAvatar";
import { AUTOPILOT_NAME } from "@/components/molecules/AutopilotAvatar/helpers";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuTrigger,
} from "@/components/molecules/DropdownMenu/DropdownMenu";
import { ExpertAvatar } from "@/components/molecules/ExpertAvatar/ExpertAvatar";
import { formatTimeAgo } from "@/lib/utils/time";
import { MoreHorizontalIcon } from "@hugeicons/core-free-icons";
import Link from "next/link";
import type { DragEvent } from "react";
import {
  type BoardColumnDef,
  getPreviewLine,
  isSessionRunning,
  TASK_CARD_MIME,
} from "../../helpers";

interface Props {
  session: SessionSummaryResponse;
  columnId: string;
  allColumns: BoardColumnDef[];
  expert: ExpertIdentity | undefined;
  onMove: (columnId: string) => void;
}

export function TaskCard({
  session,
  columnId,
  allColumns,
  expert,
  onMove,
}: Props) {
  const title = session.title || "Untitled chat";
  const isRunning = isSessionRunning(session);
  const name = expert?.name ?? AUTOPILOT_NAME;

  function handleDragStart(e: DragEvent<HTMLElement>) {
    e.dataTransfer.setData(TASK_CARD_MIME, session.id);
    e.dataTransfer.effectAllowed = "move";
  }

  return (
    <article
      draggable
      onDragStart={handleDragStart}
      className="group relative cursor-grab rounded-xl bg-white transition-shadow smooth-shadow-ring-sm hover:shadow-md active:cursor-grabbing"
    >
      <Link
        href={`/home?sessionId=${session.id}`}
        draggable={false}
        className="block rounded-xl p-3 outline-none focus-visible:ring-2 focus-visible:ring-violet-400"
      >
        <div className="flex items-center gap-1.5 pr-7">
          {expert ? (
            <ExpertAvatar
              name={expert.name}
              avatarUrl={expert.avatarUrl}
              color={expert.color}
              size={18}
              className="rounded-full"
            />
          ) : (
            <AutopilotAvatar size={18} className="rounded-full" />
          )}
          <Text variant="small-medium" tone="secondary" className="truncate">
            {name}
          </Text>
          <Text variant="small" tone="muted" className="shrink-0">
            · {isRunning ? "working" : formatTimeAgo(session.updated_at)}
          </Text>
          {isRunning && (
            <span
              aria-hidden
              className="size-1.5 shrink-0 animate-pulse rounded-full bg-violet-500"
            />
          )}
        </div>
        <Text
          variant="body-medium"
          tone="primary"
          className="mt-1.5 line-clamp-2"
        >
          {title}
        </Text>
        <Text variant="small" tone="muted" className="mt-0.5 truncate">
          {session.last_message
            ? getPreviewLine(session.last_message)
            : "Not started yet"}
        </Text>
      </Link>
      <div className="absolute right-1.5 top-1.5 opacity-0 transition-opacity focus-within:opacity-100 group-hover:opacity-100 max-md:opacity-100">
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant="ghost" size="icon-xs" aria-label={`Move ${title}`}>
              <Icon icon={MoreHorizontalIcon} size={14} />
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
                >
                  {column.name}
                </DropdownMenuItem>
              ))}
          </DropdownMenuContent>
        </DropdownMenu>
      </div>
    </article>
  );
}
