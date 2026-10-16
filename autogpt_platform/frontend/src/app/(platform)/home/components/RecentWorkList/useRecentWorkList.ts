import { useExpertMap } from "@/app/(platform)/copilot/useExpertMap";
import { useListWorkspaceFiles } from "@/app/api/__generated__/endpoints/workspace/workspace";
import { okData } from "@/app/api/helpers";
import { useHomePage } from "../../useHomePage";
import { buildRecentWorkLines } from "./helpers";

// Enough to still fill five lines after tool output and hidden files drop out.
const FILE_FETCH_LIMIT = 30;

export function useRecentWorkList() {
  const { dashboard, isLoading } = useHomePage({ enabled: true });
  const { expertsById } = useExpertMap();
  const filesQuery = useListWorkspaceFiles(
    { origin: "generated", limit: FILE_FETCH_LIMIT },
    { query: { select: (res) => okData(res)?.files ?? [] } },
  );

  return {
    isLoading: isLoading || filesQuery.isLoading,
    lines: buildRecentWorkLines(
      dashboard?.recent_work,
      filesQuery.data ?? [],
      expertsById,
    ),
  };
}
