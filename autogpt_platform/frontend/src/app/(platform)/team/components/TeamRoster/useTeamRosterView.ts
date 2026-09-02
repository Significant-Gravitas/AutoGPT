import { Expert } from "@/app/api/__generated__/models/expert";
import { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import { useState } from "react";
import { TeamFilter, filterExperts } from "../../helpers";

export type TeamView = "cards" | "table";

interface Args {
  experts: Expert[];
  schedulesForExpert: (expert: Expert) => GraphExecutionJobInfo[];
}

export function useTeamRosterView({ experts, schedulesForExpert }: Args) {
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState<TeamFilter>("all");
  const [view, setView] = useState<TeamView>("cards");

  return {
    query,
    setQuery,
    filter,
    setFilter,
    view,
    setView,
    isNarrowed: query.trim().length > 0 || filter !== "all",
    visibleExperts: filterExperts({
      experts,
      query,
      filter,
      schedulesForExpert,
    }),
  };
}
