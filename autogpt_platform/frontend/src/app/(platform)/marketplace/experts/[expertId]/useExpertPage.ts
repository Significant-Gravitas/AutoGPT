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
 *  can hire it. Templates are public, so the profile loads for everyone; the
 *  hired roster and the hire itself need a session and the experts flag.
 *  With the flag off the header shows its coming-soon label instead. */
export function useExpertPage({ expertId }: Args) {
  const { isLoggedIn, isUserLoading } = useAuth();
  const { enabled, ready } = useFlagStatus(Flag.HIRE_EXPERTS);
  const isHiringOpen = Boolean(enabled);
  const canHire = isLoggedIn && isHiringOpen;

  const templatesQuery = useListExpertTemplates({
    query: { select: (x) => x.data as Expert[] },
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
    isHiringOpen,
    // Which header action to show is only decided once LaunchDarkly has
    // answered: rendering "Coming soon" first would flash the wrong state at
    // the users who do have hiring.
    isActionReady: !isUserLoading && ready,
    isLoading: templatesQuery.isLoading,
    isError: templatesQuery.isError,
    refetch: templatesQuery.refetch,
  };
}
