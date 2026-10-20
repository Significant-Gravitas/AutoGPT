import {
  useListExperts,
  useListExpertTemplates,
} from "@/app/api/__generated__/endpoints/experts/experts";
import { useGetV1ListSystemProviders } from "@/app/api/__generated__/endpoints/integrations/integrations";
import { okData } from "@/app/api/helpers";
import { Expert } from "@/app/api/__generated__/models/expert";
import { ExpertTemplate } from "@/app/api/__generated__/models/expertTemplate";
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

  const templatesQuery = useListExpertTemplates(undefined, {
    query: { select: (x) => x.data as ExpertTemplate[] },
  });
  const expertsQuery = useListExperts({
    query: { select: (x) => x.data as Expert[], enabled: canHire },
  });
  // Which providers the platform already pays for. Public, so a signed-out
  // visitor gets it too — without it the access list cannot tell an
  // integration the viewer must connect from one they never will.
  const systemProvidersQuery = useGetV1ListSystemProviders({
    query: { select: (res) => okData(res) ?? [] },
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
    // Undefined until the list is in: the access section renders nothing
    // rather than risk naming a provider the platform supplies.
    systemProviders: systemProvidersQuery.data,
    isLoggedIn,
    isHiringOpen,
    // Which header action to show is only decided once LaunchDarkly has
    // answered and the roster is in: rendering "Coming soon" or a "Hire"
    // button first would flash the wrong state at users who have hiring, or
    // have already hired this expert. A disabled roster query never blocks —
    // isLoading is false unless it is actually fetching.
    isActionReady: !isUserLoading && ready && !expertsQuery.isLoading,
    isLoading: templatesQuery.isLoading,
    isError: templatesQuery.isError,
    refetch: templatesQuery.refetch,
  };
}
