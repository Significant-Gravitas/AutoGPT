import { useLayoutEffect, useState } from "react";
import { useHomePage } from "@/app/(platform)/home/useHomePage";
import { getInputPlaceholder } from "./helpers";

interface Args {
  enabled: boolean;
}

const TEAM_PLACEHOLDER =
  "What should the team work on? e.g. 'Follow up with yesterday's leads'";

export function useHomeComposer({ enabled }: Args) {
  const { dashboard } = useHomePage({ enabled });
  const hasExistingWork = Boolean(
    enabled &&
      dashboard &&
      (dashboard.team.total > 0 ||
        dashboard.attention.length > 0 ||
        dashboard.active_tasks.length > 0 ||
        dashboard.upcoming_tasks.length > 0 ||
        dashboard.week.run_count > 0 ||
        (dashboard.recent_work?.total_count ?? 0) > 0 ||
        (dashboard.recent_work?.groups?.length ?? 0) > 0),
  );
  const [discoveryPlaceholder, setDiscoveryPlaceholder] = useState(
    getInputPlaceholder(),
  );

  useLayoutEffect(() => {
    function handleResize() {
      setDiscoveryPlaceholder(getInputPlaceholder(window.innerWidth));
    }
    handleResize();
    const queries = [
      window.matchMedia("(max-width: 500px)"),
      window.matchMedia("(max-width: 1080px)"),
    ];
    queries.forEach((query) => query.addEventListener("change", handleResize));
    return () => {
      queries.forEach((query) =>
        query.removeEventListener("change", handleResize),
      );
    };
  }, []);

  return {
    inputPlaceholder: hasExistingWork ? TEAM_PLACEHOLDER : discoveryPlaceholder,
  };
}
