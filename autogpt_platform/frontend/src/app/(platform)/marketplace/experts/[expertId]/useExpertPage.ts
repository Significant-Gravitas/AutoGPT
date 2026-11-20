import {
  useListExperts,
  useListExpertTemplates,
} from "@/app/api/__generated__/endpoints/experts/experts";
import { useGetV1ListSystemProviders } from "@/app/api/__generated__/endpoints/integrations/integrations";
import { Expert } from "@/app/api/__generated__/models/expert";
import { ExpertTemplate } from "@/app/api/__generated__/models/expertTemplate";
import { okData } from "@/app/api/helpers";
import { useAuth } from "@/lib/auth/hooks/useAuth";

interface Args {
  expertId: string;
}

/** Public template and signed-in viewer hiring state. */
export function useExpertPage({ expertId }: Args) {
  const { isLoggedIn, isUserLoading } = useAuth();

  const canHire = isLoggedIn;

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
    isActionReady: !isUserLoading && !expertsQuery.isLoading,
    isLoading: templatesQuery.isLoading,
    isError: templatesQuery.isError,
    refetch: templatesQuery.refetch,
  };
}
