"use client";

import { ExpertAvatar } from "@/app/(platform)/copilot/components/ChatMessagesContainer/components/ExpertAvatar/ExpertAvatar";
import { Badge } from "@/components/atoms/Badge/Badge";
import { LoadingSpinner } from "@/components/atoms/LoadingSpinner/LoadingSpinner";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import type { useSortable } from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";
import type { Card } from "../../board-helpers";
import { CardFilesPopover } from "./components/CardFilesPopover/CardFilesPopover";
import { useTaskCard } from "./useTaskCard";

interface Props {
  card: Card;
  onOpen: (sessionId: string) => void;
  sortable?: ReturnType<typeof useSortable>;
  isOverlay?: boolean;
}

export function TaskCard({ card, onOpen, sortable, isOverlay = false }: Props) {
  const { title, stateLine, isRunning, needsYou, fileCount, age, assignee } =
    useTaskCard(card);

  return (
    <article
      ref={sortable?.setNodeRef}
      aria-label={title}
      style={
        sortable && {
          transform: CSS.Transform.toString(sortable.transform),
          transition: sortable.transition,
        }
      }
      className={cn(
        "group relative rounded-lg border border-zinc-200 bg-white p-3 shadow-sm transition-[border-color,box-shadow] hover:border-zinc-300 hover:shadow",
        sortable?.isDragging && "opacity-40",
        isOverlay && "cursor-grabbing border-zinc-300 shadow-lg",
      )}
    >
      {/* The drag handle is this button, whose ::after covers the card. The
          files popover sits above it and outside it, so its clicks and the
          viewer modal's never start a drag. */}
      <button
        ref={sortable?.setActivatorNodeRef}
        type="button"
        {...sortable?.attributes}
        {...sortable?.listeners}
        onClick={() => onOpen(card.id)}
        className="block w-full truncate text-left text-sm font-medium text-zinc-900 after:absolute after:inset-0 after:rounded-lg focus-visible:outline-none focus-visible:after:ring-2 focus-visible:after:ring-zinc-800"
      >
        {title}
      </button>
      {stateLine && (
        <Text variant="small" tone="muted" className="mt-0.5 truncate">
          {stateLine}
        </Text>
      )}
      <div className="mt-2.5 flex items-center gap-2">
        <div className="flex min-w-0 items-center gap-1.5">
          <ExpertAvatar
            name={assignee.name ?? ""}
            avatarUrl={assignee.avatarUrl}
            color={assignee.color}
            isAutopilot={assignee.isAutopilot}
            isLoading={assignee.name === null}
            size="sm"
          />
          {assignee.name && (
            <Text
              variant="small"
              tone="secondary"
              as="span"
              className="truncate"
            >
              {assignee.name}
            </Text>
          )}
        </div>
        <div className="ml-auto flex shrink-0 items-center gap-2 text-zinc-500">
          {fileCount > 0 && (
            <CardFilesPopover sessionId={card.id} fileCount={fileCount} />
          )}
          {isRunning && (
            <LoadingSpinner
              size="small"
              className="size-3.5 text-zinc-400"
              aria-label="Working"
            />
          )}
          {needsYou && (
            <Badge variant="warning" size="small">
              <span
                aria-hidden="true"
                className="size-1.5 rounded-full bg-amber-500"
              />
              Needs you
            </Badge>
          )}
          <Text variant="small" tone="muted" as="span" className="tabular-nums">
            {age}
          </Text>
        </div>
      </div>
    </article>
  );
}
