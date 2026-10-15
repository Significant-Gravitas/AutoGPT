import type { ExpertIdentityMap } from "@/app/(platform)/copilot/useExpertMap";
import type { SessionSummaryResponse } from "@/app/api/__generated__/models/sessionSummaryResponse";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/molecules/DropdownMenu/DropdownMenu";
import { cn } from "@/lib/utils";
import { MoreHorizontalIcon } from "@hugeicons/core-free-icons";
import type { BoardColumnDef } from "../../helpers";
import { StatusIcon } from "../StatusIcon/StatusIcon";
import { TaskCard } from "../TaskCard/TaskCard";
import { useBoardColumn } from "./useBoardColumn";

interface Props {
  column: BoardColumnDef & { sessions: SessionSummaryResponse[] };
  allColumns: BoardColumnDef[];
  expertsById: ExpertIdentityMap;
  nextFollowupBySession: Map<string, string>;
  onMoveCard: (sessionId: string, columnId: string) => void;
  onRemove?: () => void;
  onClear?: () => void;
  isClearing?: boolean;
}

export function BoardColumn({
  column,
  allColumns,
  expertsById,
  nextFollowupBySession,
  onMoveCard,
  onRemove,
  onClear,
  isClearing,
}: Props) {
  const { isOver, handleDragOver, handleDragLeave, handleDrop } =
    useBoardColumn({ columnId: column.id, onMoveCard });

  return (
    <section
      aria-label={column.name}
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      className={cn(
        "flex h-[calc(100dvh-8rem)] min-h-[32rem] min-w-[18rem] max-w-[26rem] flex-1 basis-0 flex-col rounded-xl bg-zinc-100/80 transition-colors",
        isOver && "bg-zinc-200/70",
      )}
    >
      <header className="flex h-11 items-center gap-2 pl-3.5 pr-2">
        <StatusIcon columnId={column.id} />
        <Text variant="body-medium" tone="primary" className="truncate">
          {column.name}
        </Text>
        <Text variant="body" tone="muted" className="tabular-nums">
          {column.sessions.length}
        </Text>
        <div className="ml-auto flex items-center gap-1">
          {onClear && (
            <Button
              variant="ghost"
              size="xs"
              className="text-zinc-500"
              aria-label={`Archive every chat in ${column.name}`}
              disabled={isClearing}
              onClick={onClear}
            >
              Clear
            </Button>
          )}
          {onRemove && (
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon-xs"
                  className="text-zinc-500"
                  aria-label={`${column.name} column options`}
                >
                  <Icon icon={MoreHorizontalIcon} size={16} />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end">
                <DropdownMenuItem onSelect={onRemove}>
                  Remove column
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          )}
        </div>
      </header>
      <div className="flex min-h-0 flex-1 flex-col gap-2 overflow-y-auto px-2 pb-2 scrollbar-thin scrollbar-track-transparent scrollbar-thumb-zinc-300">
        {column.sessions.map((session) => (
          <TaskCard
            key={session.id}
            session={session}
            columnId={column.id}
            allColumns={allColumns}
            expert={
              session.expert_id ? expertsById.get(session.expert_id) : undefined
            }
            nextFollowup={nextFollowupBySession.get(session.id)}
            onMove={(columnId) => onMoveCard(session.id, columnId)}
          />
        ))}
        {column.sessions.length === 0 && (
          <div className="flex items-center justify-center rounded-lg border border-dashed border-zinc-300 py-8">
            <Text variant="small" tone="muted">
              Drop a card here
            </Text>
          </div>
        )}
      </div>
    </section>
  );
}
