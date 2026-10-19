import { useExpertMap } from "@/app/(platform)/copilot/useExpertMap";
import { useSessionList } from "@/app/(platform)/copilot/useSessionList";
import {
  useGetV1ListExecutionSchedulesForAUser,
  useListCopilotFollowupSchedules,
} from "@/app/api/__generated__/endpoints/schedules/schedules";
import { okData } from "@/app/api/helpers";
import {
  buildScheduleCards,
  getNextFollowupBySession,
  groupSessionsIntoColumns,
  isDefaultColumn,
  makeColumnId,
} from "./helpers";
import { useArchiveSessions } from "./useArchiveSessions";
import { useBoardState } from "./useBoardState";

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
  const { isClearing, clearColumn } = useArchiveSessions();
  const { board, updateBoard } = useBoardState();

  function moveCard(sessionId: string, columnId: string) {
    const session = sessions.find((s) => s.id === sessionId);
    if (!session) return;
    updateBoard((prev) => ({
      ...prev,
      placements: {
        ...prev.placements,
        [sessionId]: {
          columnId,
          updatedAt: new Date(session.updated_at).toISOString(),
        },
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
