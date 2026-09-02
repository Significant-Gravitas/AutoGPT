import { useListWorkspaceFiles } from "@/app/api/__generated__/endpoints/workspace/workspace";
import { okData } from "@/app/api/helpers";

export function useTaskFilesCard(sessionId: string | null) {
  const filesQuery = useListWorkspaceFiles(
    { session_id: sessionId },
    {
      query: {
        select: (res) => okData(res) ?? null,
        enabled: Boolean(sessionId),
      },
    },
  );

  const files = filesQuery.data?.files ?? [];

  return { files };
}
