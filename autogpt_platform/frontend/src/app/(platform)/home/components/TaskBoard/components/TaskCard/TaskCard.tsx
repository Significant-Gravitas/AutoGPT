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
import { StatusIcon } from "../StatusIcon/StatusIcon";

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
  const jobTitle = expert ? expert.jobTitle || expert.role : "Head of AI";

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
        <div className="flex items-center gap-1.5">
          {expert ? (
            <ExpertAvatar
              name={expert.name}
              avatarUrl={expert.avatarUrl}
              color={expert.color}
              size={20}
              className="rounded-full border border-zinc-200"
            />
          ) : (
            <AutopilotAvatar
              size={20}
              className="rounded-full border border-zinc-200"
            />
          )}
          <Text variant="small" tone="muted" className="shrink-0">
            {name}
          </Text>
          {jobTitle && (
            <Text variant="small" className="truncate text-zinc-400">
              · {jobTitle}
            </Text>
          )}
        </div>
        <div className="mt-2.5 flex items-start gap-2">
          <StatusIcon columnId={columnId} className="mt-[0.2rem]" />
          <Text variant="body-medium" tone="primary" className="line-clamp-2">
            {title}
          </Text>
        </div>
        <Text variant="small" tone="secondary" className="mt-1 truncate pl-5">
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
        <Text variant="small" tone="muted" className="mt-2.5">
          Updated {formatTimeAgo(session.updated_at)}
        </Text>
      </Link>
      <div className="absolute bottom-2 right-2 opacity-0 transition-opacity focus-within:opacity-100 group-hover:opacity-100 max-md:opacity-100">
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
