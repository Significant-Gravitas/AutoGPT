import {
  getTeamScopedQueryKey,
  getTenantRequestInit,
} from "@/components/contextual/TeamPicker/helpers";
import { useOrgTeamStore } from "@/services/org-team/store";
import {
  getListExpertsQueryKey,
  useListExperts,
  useListExpertTemplates,
} from "@/app/api/__generated__/endpoints/experts/experts";
import { Expert } from "@/app/api/__generated__/models/expert";
import { useAuth } from "@/lib/auth/hooks/useAuth";

/** Templates are public, so the section can show them to anyone; only the
 *  hired roster (for the "Hired" state) needs a session. */
export function useExpertsSection() {
  const { activeOrgID, activeTeamID, isLoaded } = useOrgTeamStore();
  const { isLoggedIn } = useAuth();

  const templatesQuery = useListExpertTemplates({
    query: { select: (x) => x.data as Expert[] },
  });
  const expertsQuery = useListExperts({
    query: {
      select: (x) => x.data as Expert[],
      enabled: isLoggedIn && isLoaded,
      queryKey: getTeamScopedQueryKey(
        getListExpertsQueryKey(),
        activeOrgID,
        activeTeamID,
      ),
    },
    request: getTenantRequestInit(activeOrgID, activeTeamID, isLoaded),
  });

  const hiredTemplateIds = new Set<string>();
  for (const expert of expertsQuery.data ?? []) {
    if (!expert.is_archived && expert.source_template_id) {
      hiredTemplateIds.add(expert.source_template_id);
    }
  }

  return {
    isLoggedIn,
    templates: templatesQuery.data ?? [],
    hiredTemplateIds,
    isLoading: templatesQuery.isLoading,
    isError: templatesQuery.isError,
  };
}
