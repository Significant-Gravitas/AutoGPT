import {
  getTeamScopedQueryKey,
  getTenantRequestInit,
} from "@/components/contextual/TeamPicker/helpers";
import { useOrgTeamStore } from "@/services/org-team/store";
import {
  getListExpertsQueryKey,
  useListExperts,
} from "@/app/api/__generated__/endpoints/experts/experts";
import {
  getGetV2ListLibraryAgentsQueryKey,
  useGetV2ListLibraryAgentsInfinite,
} from "@/app/api/__generated__/endpoints/library/library";
import {
  getGetV1ListExecutionSchedulesForAUserQueryKey,
  useGetV1ListExecutionSchedulesForAUser,
} from "@/app/api/__generated__/endpoints/schedules/schedules";
import {
  getListCopilotSkillsQueryKey,
  useListCopilotSkills,
} from "@/app/api/__generated__/endpoints/skills/skills";
import { Expert } from "@/app/api/__generated__/models/expert";
import {
  getPaginationNextPageNumber,
  okData,
  unpaginate,
} from "@/app/api/helpers";
import {
  getAutopilotSkills,
  getAutopilotWorkflows,
  getHiredExperts,
  getTeamSchedules,
} from "../helpers";
import { useEffect } from "react";

interface Args {
  enabled: boolean;
}

export function useAutopilotPage({ enabled }: Args) {
  const { activeOrgID, activeTeamID, isLoaded } = useOrgTeamStore();
  const request = getTenantRequestInit(activeOrgID, activeTeamID, isLoaded);
  const canFetch = enabled && isLoaded;
  const libraryParams = { page: 1, page_size: 100, is_hidden: false };
  const expertsQuery = useListExperts({
    query: {
      select: (res) => (okData(res) ?? []) as Expert[],
      enabled: canFetch,
      queryKey: getTeamScopedQueryKey(
        getListExpertsQueryKey(),
        activeOrgID,
        activeTeamID,
      ),
    },
    request,
  });
  const schedulesQuery = useGetV1ListExecutionSchedulesForAUser({
    query: {
      select: (res) => okData(res) ?? [],
      enabled: canFetch,
      queryKey: getTeamScopedQueryKey(
        getGetV1ListExecutionSchedulesForAUserQueryKey(),
        activeOrgID,
        activeTeamID,
      ),
    },
    request,
  });
  const libraryQuery = useGetV2ListLibraryAgentsInfinite(libraryParams, {
    query: {
      getNextPageParam: getPaginationNextPageNumber,
      enabled: canFetch,
      queryKey: getTeamScopedQueryKey(
        [...getGetV2ListLibraryAgentsQueryKey(libraryParams), "infinite"],
        activeOrgID,
        activeTeamID,
      ),
    },
    request,
  });
  const skillsQuery = useListCopilotSkills({
    query: {
      select: (res) => okData(res) ?? [],
      enabled: canFetch,
      queryKey: getTeamScopedQueryKey(
        getListCopilotSkillsQueryKey(),
        activeOrgID,
        activeTeamID,
      ),
    },
    request,
  });

  // Autopilot owns every library workflow no expert has claimed, so the
  // whole library has to be in hand before the split is derived.
  const { hasNextPage, isFetchingNextPage, fetchNextPage } = libraryQuery;
  useEffect(() => {
    if (hasNextPage && !isFetchingNextPage) void fetchNextPage();
  }, [hasNextPage, isFetchingNextPage, fetchNextPage]);

  const experts = getHiredExperts(expertsQuery.data ?? []);
  const allSchedules = schedulesQuery.data ?? [];
  const schedules = getTeamSchedules(experts, allSchedules);
  const libraryAgents = libraryQuery.data
    ? unpaginate(libraryQuery.data, "agents")
    : [];
  const workflows = getAutopilotWorkflows(experts, libraryAgents, allSchedules);
  const skills = getAutopilotSkills(experts, skillsQuery.data ?? []);

  function refetch() {
    return Promise.all([
      expertsQuery.refetch(),
      schedulesQuery.refetch(),
      libraryQuery.refetch(),
      skillsQuery.refetch(),
    ]);
  }

  return {
    experts,
    schedules,
    workflows,
    skills,
    isLoading:
      enabled &&
      (expertsQuery.isLoading ||
        schedulesQuery.isLoading ||
        libraryQuery.isLoading ||
        skillsQuery.isLoading),
    isError: expertsQuery.isError || schedulesQuery.isError,
    refetch,
  };
}
