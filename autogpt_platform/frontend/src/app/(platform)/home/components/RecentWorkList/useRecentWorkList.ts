import { useExpertMap } from "@/app/(platform)/copilot/useExpertMap";
import { useListWorkspaceFiles } from "@/app/api/__generated__/endpoints/workspace/workspace";
import { okData } from "@/app/api/helpers";
import { useHomePage } from "../../useHomePage";
import { buildRecentWorkLines } from "./helpers";

// Enough to still fill five lines after tool output and hidden files drop out.
const FILE_FETCH_LIMIT = 30;

export function useRecentWorkList() {
  const { dashboard, isLoading } = useHomePage({ enabled: true });
  const { expertsById, isLoadingExperts, hasExpertsErrored } = useExpertMap();
  const filesQuery = useListWorkspaceFiles(
    { origin: "generated", limit: FILE_FETCH_LIMIT },
    { query: { select: (res) => okData(res)?.files ?? [] } },
  );

  // Until the roster arrives, an expert's file would be credited to Otto.
  // A background refetch keeps the roster it already has, so only the first
  // load waits; a failed roster drops file lines and keeps the rest.
  const hasNoRoster = expertsById.size === 0;

  return {
    isLoading:
      isLoading || filesQuery.isLoading || (isLoadingExperts && hasNoRoster),
    lines: buildRecentWorkLines(
      dashboard?.recent_work,
      hasExpertsErrored && hasNoRoster ? [] : (filesQuery.data ?? []),
      expertsById,
    ),
  };
}
