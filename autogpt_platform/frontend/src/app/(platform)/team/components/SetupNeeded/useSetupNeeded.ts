import {
  getListExpertSetupItemsQueryKey,
  getListExpertsQueryKey,
  useGrantExpertCredentials,
  useListExpertSetupItems,
} from "@/app/api/__generated__/endpoints/experts/experts";
import { useGetV1ListProviders } from "@/app/api/__generated__/endpoints/integrations/integrations";
import { getGetV1ListExecutionSchedulesForAUserQueryKey } from "@/app/api/__generated__/endpoints/schedules/schedules";
import type { CredentialsMetaResponse } from "@/app/api/__generated__/models/credentialsMetaResponse";
import type { ExpertSetupItem } from "@/app/api/__generated__/models/expertSetupItem";
import { okData } from "@/app/api/helpers";
import { toConnectableProviders } from "@/components/contextual/IntegrationsPanel/components/ConnectServiceDialog/helpers";
import { formatProviderName } from "@/components/contextual/IntegrationsPanel/helpers";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

interface Args {
  enabled: boolean;
}

export function useSetupNeeded({ enabled }: Args) {
  const queryClient = useQueryClient();
  const { toast } = useToast();
  const [connecting, setConnecting] = useState<ExpertSetupItem | null>(null);

  const itemsQuery = useListExpertSetupItems({
    query: { enabled, select: (res) => okData(res) ?? [] },
  });
  const providersQuery = useGetV1ListProviders({
    query: {
      enabled,
      select: (res) => (res.status === 200 ? res.data : []),
    },
  });
  const connectable = new Set(
    toConnectableProviders(providersQuery.data ?? []).map((p) => p.id),
  );

  // A grant is what lets the backend create the pending schedule, so the
  // roster (schedule counts) and the schedule list move with the card.
  function invalidate() {
    for (const queryKey of [
      getListExpertSetupItemsQueryKey(),
      getListExpertsQueryKey(),
      getGetV1ListExecutionSchedulesForAUserQueryKey(),
    ]) {
      queryClient.invalidateQueries({ queryKey });
    }
  }

  const { mutate: grant, isPending: isGranting } = useGrantExpertCredentials({
    mutation: {
      onSuccess: invalidate,
      onError: () =>
        toast({ title: "Could not add integration", variant: "destructive" }),
    },
  });

  function grantTo(
    item: ExpertSetupItem,
    credential: { id: string; provider: string },
  ) {
    grant(
      { expertId: item.expert_id, data: { credential_ids: [credential.id] } },
      {
        onSuccess: () =>
          toast({
            title: `${item.expert_name} can now use ${formatProviderName(credential.provider)}`,
          }),
      },
    );
  }

  function allow(item: ExpertSetupItem) {
    if (!item.credential_id) return;
    grantTo(item, {
      id: item.credential_id,
      provider: item.providers[0] ?? "",
    });
  }

  function connect(item: ExpertSetupItem) {
    setConnecting(item);
  }

  function closeConnect() {
    setConnecting(null);
  }

  function handleConnected(credential: CredentialsMetaResponse) {
    if (connecting) grantTo(connecting, credential);
    setConnecting(null);
  }

  return {
    items: itemsQuery.data ?? [],
    // While the provider list is still loading nothing is knowably
    // unconnectable, so rows offer Connect rather than flashing "Needs a
    // platform key". Once it has settled, a failed load means no provider
    // is known to be connectable and the rows say so.
    isConnectable: (item: ExpertSetupItem) =>
      providersQuery.isPending ||
      item.providers.some((provider) => connectable.has(provider)),
    connecting,
    connect,
    closeConnect,
    handleConnected,
    allow,
    isGranting,
  };
}
