import { useExpertMap } from "@/app/(platform)/copilot/useExpertMap";
import {
  SESSION_LIST_QUERY_KEY,
  useSessionList,
} from "@/app/(platform)/copilot/useSessionList";
import { usePostV2ArchiveSessions } from "@/app/api/__generated__/endpoints/chat/chat";
import {
  useGetV1ListExecutionSchedulesForAUser,
  useListCopilotFollowupSchedules,
} from "@/app/api/__generated__/endpoints/schedules/schedules";
import { okData } from "@/app/api/helpers";
import { Button } from "@/components/atoms/Button/Button";
import { toast } from "@/components/molecules/Toast/use-toast";
import { Key, storage } from "@/services/storage/local-storage";
import { useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import {
  type BoardState,
  buildScheduleCards,
  getNextFollowupBySession,
  groupSessionsIntoColumns,
  isDefaultColumn,
  makeColumnId,
  parseBoardState,
} from "./helpers";

const SCHEDULES_REFETCH_INTERVAL_MS = 60_000;

export function useTaskBoard() {
  const { sessions, isLoading } = useSessionList();
  const { expertsById } = useExpertMap();
  const { data: followups = [] } = useListCopilotFollowupSchedules({
    query: {
      select: (res) => okData(res) ?? [],
      refetchInterval: SCHEDULES_REFETCH_INTERVAL_MS,
    },
  });
  const { data: workflowSchedules = [] } =
    useGetV1ListExecutionSchedulesForAUser({
      query: {
        select: (res) => okData(res) ?? [],
        refetchInterval: SCHEDULES_REFETCH_INTERVAL_MS,
      },
    });
  const queryClient = useQueryClient();
  const { mutateAsync: setArchived, isPending: isClearing } =
    usePostV2ArchiveSessions();
  const [board, setBoard] = useState(() =>
    parseBoardState(storage.get(Key.EXPERT_TASK_BOARD)),
  );

  function updateBoard(update: (prev: BoardState) => BoardState) {
    setBoard((prev) => {
      const next = update(prev);
      storage.set(Key.EXPERT_TASK_BOARD, JSON.stringify(next));
      return next;
    });
  }

  function moveCard(sessionId: string, columnId: string) {
    const session = sessions.find((s) => s.id === sessionId);
    if (!session) return;
    updateBoard((prev) => ({
      ...prev,
      placements: {
        ...prev.placements,
        [sessionId]: { columnId, updatedAt: session.updated_at },
      },
    }));
  }

  function addColumn(name: string) {
    const trimmed = name.trim();
    if (!trimmed) return;
    updateBoard((prev) => ({
      ...prev,
      columns: [...prev.columns, { id: makeColumnId(trimmed), name: trimmed }],
    }));
  }

  function removeColumn(columnId: string) {
    if (isDefaultColumn(columnId)) return;
    updateBoard((prev) => ({
      ...prev,
      columns: prev.columns.filter((column) => column.id !== columnId),
    }));
  }

  async function archiveSessions(sessionIds: string[], archived: boolean) {
    try {
      await setArchived({ data: { session_ids: sessionIds, archived } });
      return true;
    } catch {
      toast({
        title: archived
          ? "Could not archive these chats"
          : "Could not restore these chats",
        variant: "destructive",
      });
      return false;
    } finally {
      queryClient.invalidateQueries({ queryKey: SESSION_LIST_QUERY_KEY });
    }
  }

  async function clearColumn(sessionIds: string[]) {
    if (!sessionIds.length || !(await archiveSessions(sessionIds, true))) {
      return;
    }
    const count = sessionIds.length;
    const shown = toast({
      title: `${count} ${count === 1 ? "chat" : "chats"} archived`,
      duration: 8000,
      action: (
        <Button
          variant="secondary"
          size="xs"
          className="ml-auto"
          onClick={() => {
            shown.dismiss();
            void archiveSessions(sessionIds, false);
          }}
        >
          Undo
        </Button>
      ),
    });
  }

  return {
    isLoading,
    isClearing,
    clearColumn,
    columns: groupSessionsIntoColumns(sessions, board),
    schedules: buildScheduleCards(followups, workflowSchedules),
    nextFollowupBySession: getNextFollowupBySession(followups),
    allColumns: board.columns,
    expertsById,
    moveCard,
    addColumn,
    removeColumn,
  };
}
