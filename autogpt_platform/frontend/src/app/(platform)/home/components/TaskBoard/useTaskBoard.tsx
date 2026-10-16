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
import { useQueryClient } from "@tanstack/react-query";
import {
  buildScheduleCards,
  getNextFollowupBySession,
  groupSessionsIntoColumns,
  isDefaultColumn,
  makeColumnId,
} from "./helpers";
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
  const queryClient = useQueryClient();
  const { mutateAsync: setArchived, isPending: isClearing } =
    usePostV2ArchiveSessions();
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

  async function archiveSessions(sessionIds: string[], archived: boolean) {
    const completed: string[] = [];
    try {
      for (let offset = 0; offset < sessionIds.length; offset += 100) {
        const batch = sessionIds.slice(offset, offset + 100);
        await setArchived({ data: { session_ids: batch, archived } });
        completed.push(...batch);
      }
    } catch {
      toast({
        title: archived
          ? "Could not archive these chats"
          : "Could not restore these chats",
        variant: "destructive",
      });
    } finally {
      queryClient.invalidateQueries({ queryKey: SESSION_LIST_QUERY_KEY });
    }
    return completed;
  }

  async function clearColumn(sessionIds: string[]) {
    if (!sessionIds.length) return;
    const archivedIds = await archiveSessions(sessionIds, true);
    const count = archivedIds.length;
    if (!count) return;
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
            void archiveSessions(archivedIds, false);
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
