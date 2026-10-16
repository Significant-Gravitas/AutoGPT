import { useHomePage } from "../../useHomePage";
import { buildRecentWorkLines } from "./helpers";

export function useRecentWorkList() {
  const { dashboard, isLoading } = useHomePage({ enabled: true });

  return {
    isLoading,
    lines: buildRecentWorkLines(dashboard?.recent_work),
  };
}
