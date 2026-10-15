import type { ExpertIdentityMap } from "@/app/(platform)/copilot/useExpertMap";
import type { SessionSummaryResponse } from "@/app/api/__generated__/models/sessionSummaryResponse";
import { Button } from "@/components/atoms/Button/Button";
import { Icon } from "@/components/atoms/Icon/Icon";
import { Text } from "@/components/atoms/Text/Text";
import { cn } from "@/lib/utils";
import { Cancel01Icon } from "@hugeicons/core-free-icons";
import type { BoardColumnDef } from "../../helpers";
import { TaskCard } from "../TaskCard/TaskCard";
import { useBoardColumn } from "./useBoardColumn";

interface Props {
  column: BoardColumnDef & { sessions: SessionSummaryResponse[] };
  allColumns: BoardColumnDef[];
  expertsById: ExpertIdentityMap;
  onMoveCard: (sessionId: string, columnId: string) => void;
  onRemove?: () => void;
}

export function BoardColumn({
  column,
  allColumns,
  expertsById,
  onMoveCard,
  onRemove,
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
        "flex w-[17rem] shrink-0 flex-col rounded-2xl bg-zinc-100/70 p-2 transition-colors",
        isOver && "bg-violet-50 ring-1 ring-violet-200",
      )}
    >
      <header className="flex h-8 items-center gap-2 px-2">
        <Text variant="body-medium" tone="primary" className="truncate">
          {column.name}
        </Text>
        <Text variant="small" tone="muted" className="tabular-nums">
          {column.sessions.length}
        </Text>
        {onRemove && (
          <Button
            variant="ghost"
            size="icon-xs"
            className="ml-auto"
            aria-label={`Remove ${column.name} column`}
            onClick={onRemove}
          >
            <Icon icon={Cancel01Icon} size={14} />
          </Button>
        )}
      </header>
      <div className="flex max-h-[32rem] min-h-24 flex-col gap-2 overflow-y-auto p-0.5">
        {column.sessions.map((session) => (
          <TaskCard
            key={session.id}
            session={session}
            columnId={column.id}
            allColumns={allColumns}
            expert={
              session.expert_id ? expertsById.get(session.expert_id) : undefined
            }
            onMove={(columnId) => onMoveCard(session.id, columnId)}
          />
        ))}
        {column.sessions.length === 0 && (
          <Text variant="small" tone="muted" className="px-2 py-6 text-center">
            Drop a card here
          </Text>
        )}
      </div>
    </section>
  );
}
