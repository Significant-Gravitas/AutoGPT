import {
  useListExperts,
  useListExpertTemplates,
} from "@/app/api/__generated__/endpoints/experts/experts";
import { Expert } from "@/app/api/__generated__/models/expert";
import { useAuth } from "@/lib/auth/hooks/useAuth";
import { Flag, useFlagStatus } from "@/services/feature-flags/use-get-flag";

interface Args {
  expertId: string;
}

/** The template behind a marketplace expert page, plus whether this viewer
 *  can hire it. Templates only exist behind a signed-in session and the
 *  experts flag; without both, the page shows its coming-soon face. */
export function useExpertPage({ expertId }: Args) {
  const { isLoggedIn } = useAuth();
  const { enabled, ready } = useFlagStatus(Flag.HIRE_EXPERTS);
  const canHire = isLoggedIn && Boolean(enabled);

  const templatesQuery = useListExpertTemplates({
    query: { select: (x) => x.data as Expert[], enabled: canHire },
  });
  const expertsQuery = useListExperts({
    query: { select: (x) => x.data as Expert[], enabled: canHire },
  });

  const expert =
    (templatesQuery.data ?? []).find((template) => template.id === expertId) ??
    null;
  const hiredExpert =
    (expertsQuery.data ?? []).find(
      (hired) => !hired.is_archived && hired.source_template_id === expertId,
    ) ?? null;

  return {
    expert,
    hiredExpert,
    canHire,
    isReady: ready,
    isLoading: canHire && templatesQuery.isLoading,
    isError: templatesQuery.isError,
    refetch: templatesQuery.refetch,
  };
}
