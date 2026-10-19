import { SESSION_LIST_QUERY_KEY } from "@/app/(platform)/copilot/useSessionList";
import { usePostV2ArchiveSessions } from "@/app/api/__generated__/endpoints/chat/chat";
import { Button } from "@/components/atoms/Button/Button";
import { toast } from "@/components/molecules/Toast/use-toast";
import { useQueryClient } from "@tanstack/react-query";

// The archive endpoint takes at most this many ids per request.
const ARCHIVE_BATCH_SIZE = 100;

export function useArchiveSessions() {
  const queryClient = useQueryClient();
  const { mutateAsync: setArchived, isPending: isClearing } =
    usePostV2ArchiveSessions();

  async function archiveSessions(sessionIds: string[], archived: boolean) {
    const completed: string[] = [];
    try {
      for (
        let offset = 0;
        offset < sessionIds.length;
        offset += ARCHIVE_BATCH_SIZE
      ) {
        const batch = sessionIds.slice(offset, offset + ARCHIVE_BATCH_SIZE);
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

  return { isClearing, clearColumn };
}
