import { Expert } from "@/app/api/__generated__/models/expert";
import { GraphExecutionJobInfo } from "@/app/api/__generated__/models/graphExecutionJobInfo";
import { useState } from "react";
import { TeamFilter, filterExperts } from "../../helpers";

interface Args {
  experts: Expert[];
  schedulesForExpert: (expert: Expert) => GraphExecutionJobInfo[];
}

export function useTeamRosterView({ experts, schedulesForExpert }: Args) {
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState<TeamFilter>("all");

  return {
    query,
    setQuery,
    filter,
    setFilter,
    isNarrowed: query.trim().length > 0 || filter !== "all",
    visibleExperts: filterExperts({
      experts,
      query,
      filter,
      schedulesForExpert,
    }),
  };
}
