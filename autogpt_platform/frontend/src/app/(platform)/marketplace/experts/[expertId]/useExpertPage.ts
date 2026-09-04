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
 *  can hire it. Templates are public, so a signed-out visitor sees the
 *  profile with a sign-up prompt; the hired roster and the hire itself need
 *  a session and the experts flag. Signed in without the flag, the page
 *  shows its coming-soon face. */
export function useExpertPage({ expertId }: Args) {
  const { isLoggedIn, isUserLoading } = useAuth();
  const { enabled, ready } = useFlagStatus(Flag.HIRE_EXPERTS);
  const canHire = isLoggedIn && Boolean(enabled);
  const canView = !isUserLoading && (!isLoggedIn || canHire);

  const templatesQuery = useListExpertTemplates({
    query: { select: (x) => x.data as Expert[], enabled: canView },
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
    isLoggedIn,
    isComingSoon: isLoggedIn && !enabled,
    isReady: !isUserLoading && (!isLoggedIn || ready),
    isLoading: canView && templatesQuery.isLoading,
    isError: templatesQuery.isError,
    refetch: templatesQuery.refetch,
  };
}
