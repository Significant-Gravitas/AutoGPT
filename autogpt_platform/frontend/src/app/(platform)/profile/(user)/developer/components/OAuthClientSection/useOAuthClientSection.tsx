"use client";
import {
  getGetOauthClientsListClientsQueryKey,
  useDeleteOauthClientsDeleteClient,
  useGetOauthClientsListClients,
  usePostOauthClientsActivateClient,
  usePostOauthClientsSuspendClient,
} from "@/app/api/__generated__/endpoints/oauth-clients/oauth-clients";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { getQueryClient } from "@/lib/react-query/queryClient";

export function useOAuthClientSection() {
  const queryClient = getQueryClient();
  const { toast } = useToast();

  const { data: oauthClients, isLoading } = useGetOauthClientsListClients({
    query: {
      select: (res) => {
        if (res.status !== 200) return undefined;
        return res.data;
      },
    },
  });

  const { mutateAsync: deleteClient, isPending: isDeleting } =
    useDeleteOauthClientsDeleteClient({
      mutation: {
        onSettled: () => {
          return queryClient.invalidateQueries({
            queryKey: getGetOauthClientsListClientsQueryKey(),
          });
        },
      },
    });

  const { mutateAsync: suspendClient, isPending: isSuspending } =
    usePostOauthClientsSuspendClient({
      mutation: {
        onSettled: () => {
          return queryClient.invalidateQueries({
            queryKey: getGetOauthClientsListClientsQueryKey(),
          });
        },
      },
    });

  const { mutateAsync: activateClient, isPending: isActivating } =
    usePostOauthClientsActivateClient({
      mutation: {
        onSettled: () => {
          return queryClient.invalidateQueries({
            queryKey: getGetOauthClientsListClientsQueryKey(),
          });
        },
      },
    });

  async function handleDeleteClient(clientId: string) {
    try {
      await deleteClient({ clientId });
      toast({
        title: "Success",
        description: "OAuth client deleted successfully",
      });
    } catch {
      toast({
        title: "Error",
        description: "Failed to delete OAuth client",
        variant: "destructive",
      });
    }
  }

  async function handleSuspendClient(clientId: string) {
    try {
      await suspendClient({ clientId });
      toast({
        title: "Success",
        description: "OAuth client suspended successfully",
      });
    } catch {
      toast({
        title: "Error",
        description: "Failed to suspend OAuth client",
        variant: "destructive",
      });
    }
  }

  async function handleActivateClient(clientId: string) {
    try {
      await activateClient({ clientId });
      toast({
        title: "Success",
        description: "OAuth client activated successfully",
      });
    } catch {
      toast({
        title: "Error",
        description: "Failed to activate OAuth client",
        variant: "destructive",
      });
    }
  }

  return {
    oauthClients,
    isLoading,
    isDeleting,
    isSuspending,
    isActivating,
    handleDeleteClient,
    handleSuspendClient,
    handleActivateClient,
  };
}
