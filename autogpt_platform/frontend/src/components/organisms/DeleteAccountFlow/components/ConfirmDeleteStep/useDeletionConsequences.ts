import {
  useGetV2ListMySkillSubmissions,
  useGetV2ListMySubmissions,
} from "@/app/api/__generated__/endpoints/store/store";
import { useConnectedApps } from "../../useConnectedApps";
import { usePlanContext } from "../../usePlanContext";
import { getConsequences, getPlanLabel } from "./helpers";

export function useDeletionConsequences() {
  const { apps } = useConnectedApps();
  const { tier } = usePlanContext();
  const { data: publishedWorkflows = false } = useGetV2ListMySubmissions(
    { page: 1, page_size: 1 },
    {
      query: {
        select: (response) =>
          response.status === 200 && response.data.submissions.length > 0,
      },
    },
  );
  const { data: publishedSkills = false } = useGetV2ListMySkillSubmissions({
    query: {
      select: (response) => response.status === 200 && response.data.length > 0,
    },
  });

  return getConsequences({
    appNames: apps.status === "ready" ? apps.names : [],
    planLabel: getPlanLabel(tier),
    publishedWorkflows,
    publishedSkills,
  });
}
