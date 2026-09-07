import {
  getGetV2GetMarketplaceSkillQueryKey,
  usePostV2InstallMarketplaceSkill,
} from "@/app/api/__generated__/endpoints/store/store";
import {
  getGetV1ListCredentialsQueryKey,
  useGetV1ListCredentials,
} from "@/app/api/__generated__/endpoints/integrations/integrations";
import { formatProviderName } from "@/components/contextual/IntegrationsPanel/helpers";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { useAuth } from "@/lib/auth/hooks/useAuth";
import { useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

interface Args {
  slug: string;
  requiredProviders: string[];
}

export function useInstallSkillPanel({ slug, requiredProviders }: Args) {
  const queryClient = useQueryClient();
  const { isLoggedIn } = useAuth();
  const { toast } = useToast();
  const [installedName, setInstalledName] = useState<string | null>(null);
  const [isConnectOpen, setIsConnectOpen] = useState(false);

  const credentialsQuery = useGetV1ListCredentials({
    query: {
      enabled: isLoggedIn,
      select: (response) => (response.status === 200 ? response.data : []),
    },
  });
  const connected = new Set(
    (credentialsQuery.data ?? []).map((credential) => credential.provider),
  );

  const { mutate: install, isPending } = usePostV2InstallMarketplaceSkill({
    mutation: {
      onSuccess: (response) => {
        if (response.status !== 200) return;
        setInstalledName(response.data.name);
        queryClient.invalidateQueries({
          queryKey: getGetV2GetMarketplaceSkillQueryKey(slug),
        });
      },
      onError: (error) =>
        toast({
          title: "Couldn't add this skill",
          description:
            error instanceof Error ? error.message : "Please try again.",
          variant: "destructive",
        }),
    },
  });

  function addToAutoPilot() {
    install({ slug });
  }

  function handleConnected() {
    setIsConnectOpen(false);
    queryClient.invalidateQueries({
      queryKey: getGetV1ListCredentialsQueryKey(),
    });
  }

  return {
    isLoggedIn,
    providerNames: requiredProviders.map(formatProviderName),
    installedName,
    isInstalling: isPending,
    addToAutoPilot,
    // Connecting is a next step, never a precondition: the list is only ever
    // used to offer what is still worth setting up, after the install landed.
    pendingConnections: requiredProviders
      .filter((provider) => !connected.has(provider))
      .map(formatProviderName),
    isConnectOpen,
    openConnect: () => setIsConnectOpen(true),
    setIsConnectOpen,
    handleConnected,
  };
}
