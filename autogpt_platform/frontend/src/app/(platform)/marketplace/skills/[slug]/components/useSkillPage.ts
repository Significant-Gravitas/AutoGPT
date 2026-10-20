import {
  getGetExpertQueryKey,
  useListExperts,
} from "@/app/api/__generated__/endpoints/experts/experts";
import {
  getGetV1ListCredentialsQueryKey,
  useGetV1ListCredentials,
} from "@/app/api/__generated__/endpoints/integrations/integrations";
import {
  getListCopilotSkillsQueryKey,
  useListCopilotSkills,
} from "@/app/api/__generated__/endpoints/skills/skills";
import {
  useGetV2GetMarketplaceSkill,
  useGetV2ListMarketplaceSkills,
  usePostV2InstallMarketplaceSkill,
} from "@/app/api/__generated__/endpoints/store/store";
import { Expert } from "@/app/api/__generated__/models/expert";
import { okData } from "@/app/api/helpers";
import { formatProviderName } from "@/components/contextual/IntegrationsPanel/helpers";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { useAuth } from "@/lib/auth/hooks/useAuth";
import { Flag, useFlagStatus } from "@/services/feature-flags/use-get-flag";
import { useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

/** Two others, so the shelf under the body is one row. */
const MORE_SKILLS_PAGE_SIZE = 3;

export function useSkillPage(slug: string) {
  const queryClient = useQueryClient();
  const { isLoggedIn, isUserLoading } = useAuth();
  const { toast } = useToast();
  const flag = useFlagStatus(Flag.SKILLS_HUB);
  const [isConnectOpen, setIsConnectOpen] = useState(false);
  // Page state rather than the list's: a relative link in the body opens the
  // same viewer the package list does.
  const [openFilePath, setOpenFilePath] = useState<string | null>(null);

  const query = useGetV2GetMarketplaceSkill(slug, {
    query: { select: (response) => okData(response) ?? null },
  });

  // The install lands under the listing slug, so the user's own skill names
  // are the truth about whether this one is already added — a reload no
  // longer offers to install it again.
  const installedSkills = useListCopilotSkills(undefined, {
    query: { select: (res) => okData(res) ?? [], enabled: isLoggedIn },
  });
  const isAdded = (installedSkills.data ?? []).some(
    (skill) => skill.name === slug,
  );

  const expertsQuery = useListExperts({
    query: { select: (res) => res.data as Expert[], enabled: isLoggedIn },
  });
  const experts = (expertsQuery.data ?? []).filter(
    (expert) => !expert.is_template && !expert.is_archived,
  );

  const credentials = useGetV1ListCredentials({
    query: {
      enabled: isLoggedIn,
      select: (response) => okData(response) ?? [],
    },
  });
  const connected = new Set(
    (credentials.data ?? []).map((credential) => credential.provider),
  );

  const more = useGetV2ListMarketplaceSkills(
    { page_size: MORE_SKILLS_PAGE_SIZE },
    { query: { select: (response) => okData(response)?.skills ?? [] } },
  );

  const { mutate: install, isPending: isAdding } =
    usePostV2InstallMarketplaceSkill({
      mutation: {
        onSuccess: (response, variables) => {
          if (response.status !== 200) return;
          const expert = findExpert(variables.params?.expert_id);
          queryClient.invalidateQueries({
            queryKey: getListCopilotSkillsQueryKey(
              expert ? { expert_id: expert.id } : undefined,
            ),
          });
          if (!expert) return;
          queryClient.invalidateQueries({
            queryKey: getGetExpertQueryKey(expert.id),
          });
          toast({ title: `Installed on ${expert.name}`, variant: "success" });
        },
        onError: (error, variables) => {
          const status = (error as { status?: number }).status;
          const expert = findExpert(variables.params?.expert_id);
          toast({
            title: expert
              ? `Couldn't add this skill to ${expert.name}`
              : "Couldn't add this skill",
            description:
              status === 401
                ? "Sign in and try again."
                : error instanceof Error
                  ? error.message
                  : "Please try again.",
            variant: "destructive",
          });
        },
      },
    });

  function findExpert(expertId: string | null | undefined) {
    return experts.find((expert) => expert.id === expertId) ?? null;
  }

  const requiredProviders = query.data?.required_providers ?? [];

  return {
    skill: query.data ?? null,
    isLoggedIn,
    isLoading: query.isLoading,
    isError: query.isError,
    isNotFound: (query.error as { status?: number } | null)?.status === 404,
    refetch: query.refetch,
    // Nothing renders before auth, the flag, the user's own skills and the
    // roster have all answered, so the button never flashes into "Added" nor
    // grows its dropdown half under the pointer.
    isReady:
      !isUserLoading &&
      flag.ready &&
      (!isLoggedIn || (!installedSkills.isLoading && !expertsQuery.isLoading)),
    flagEnabled: flag.enabled,
    flagReady: flag.ready,
    isAdded,
    isAdding,
    experts,
    addToAutoPilot: () => install({ slug }),
    addToExpert: (expert: Expert) =>
      install({ slug, params: { expert_id: expert.id } }),
    // Connecting is a next step, never a precondition: the list only offers
    // what is still worth setting up, after the install landed.
    pendingConnections: requiredProviders
      .filter((provider) => !connected.has(provider))
      .map(formatProviderName),
    moreSkills: (more.data ?? []).filter((skill) => skill.slug !== slug),
    files: query.data?.files ?? [],
    openFilePath,
    openFile: (path: string) => setOpenFilePath(path),
    closeFile: () => setOpenFilePath(null),
    isConnectOpen,
    openConnect: () => setIsConnectOpen(true),
    setIsConnectOpen,
    handleConnected: () => {
      setIsConnectOpen(false);
      queryClient.invalidateQueries({
        queryKey: getGetV1ListCredentialsQueryKey(),
      });
    },
  };
}
