import {
  useGrantExpertCredentials,
  useListExpertSetupItems,
  useListExperts,
  useListExpertTemplates,
} from "@/app/api/__generated__/endpoints/experts/experts";
import { useGetV1ListProviders } from "@/app/api/__generated__/endpoints/integrations/integrations";
import type { CredentialsMetaResponse } from "@/app/api/__generated__/models/credentialsMetaResponse";
import type { Expert } from "@/app/api/__generated__/models/expert";
import { okData } from "@/app/api/helpers";
import {
  getHiredExperts,
  isSettingUp,
  SETUP_POLL_MS,
} from "@/app/(platform)/team/helpers";
import { toConnectableProviders } from "@/components/contextual/IntegrationsPanel/components/ConnectServiceDialog/helpers";
import { useToast } from "@/components/molecules/Toast/use-toast";
import { invalidateExpertGrantQueries } from "@/services/experts/invalidate-experts";
import { requestComposerVoice } from "@/services/onboarding/composer-voice-request";
import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { buildTeam, chooseAsk, joinNames } from "./helpers";

const COMPOSER_INPUT_ID = "chat-input-empty";

export function useIntroStandup({ path }: { path: string }) {
  const queryClient = useQueryClient();
  const { toast } = useToast();
  const [connectDismissed, setConnectDismissed] = useState(false);
  const [isConnectOpen, setIsConnectOpen] = useState(false);

  const hiresQuery = useListExperts({
    query: {
      select: (res) => getHiredExperts((okData(res) ?? []) as Expert[]),
      refetchInterval: (query) =>
        isSettingUp((okData(query.state.data) ?? []) as Expert[])
          ? SETUP_POLL_MS
          : false,
    },
  });
  const templatesQuery = useListExpertTemplates(undefined, {
    query: { select: (res) => okData(res) ?? [] },
  });
  const setupQuery = useListExpertSetupItems({
    query: { select: (res) => okData(res) ?? [] },
  });
  const providersQuery = useGetV1ListProviders({
    query: { select: (res) => (res.status === 200 ? res.data : []) },
  });
  const connectable = new Set(
    toConnectableProviders(providersQuery.data ?? []).map((p) => p.id),
  );
  const { mutateAsync: grant } = useGrantExpertCredentials();

  // A hire's schedules, and so its setup gaps, exist only once its install
  // has finished; re-read them whenever the set of installing hires changes.
  const installing = (hiresQuery.data ?? [])
    .filter((expert) => expert.setup_status === "installing")
    .map((expert) => expert.id)
    .join(",");
  const lastInstalling = useRef<string | null>(null);
  const refetchSetup = setupQuery.refetch;
  useEffect(() => {
    if (!hiresQuery.isFetched) return;
    if (
      lastInstalling.current !== null &&
      lastInstalling.current !== installing
    )
      void refetchSetup();
    lastInstalling.current = installing;
  }, [installing, hiresQuery.isFetched, refetchSetup]);

  const setupItems = setupQuery.data ?? [];
  const team = buildTeam({
    hires: hiresQuery.data ?? [],
    templates: templatesQuery.data ?? [],
    setupItems,
  });
  // Nothing is asked until every source has answered, so the ask never
  // switches shape under the user's cursor.
  const isSettled =
    hiresQuery.isFetched &&
    setupQuery.isFetched &&
    templatesQuery.isFetched &&
    !providersQuery.isPending;
  const ask = isSettled
    ? chooseAsk({
        path,
        team,
        setupItems,
        isConnectable: (provider) => connectable.has(provider),
        connectDismissed,
      })
    : null;

  async function handleConnected(credential: CredentialsMetaResponse) {
    setIsConnectOpen(false);
    if (ask?.kind !== "connect") return;
    const results = await Promise.allSettled(
      ask.expertIds.map((expertId) =>
        grant({ expertId, data: { credential_ids: [credential.id] } }).then(
          () => invalidateExpertGrantQueries(queryClient, expertId),
        ),
      ),
    );
    if (results.some((result) => result.status === "rejected")) {
      toast({
        title: `Couldn't give ${joinNames(ask.names)} access to ${ask.service}`,
        description: "You can finish it on the Team page.",
        variant: "destructive",
      });
    }
  }

  function focusComposer() {
    document.getElementById(COMPOSER_INPUT_ID)?.focus();
  }

  return {
    team,
    ask,
    isConnectOpen,
    openConnect: () => setIsConnectOpen(true),
    closeConnect: () => setIsConnectOpen(false),
    handleConnected,
    dismissConnect: () => setConnectDismissed(true),
    startVoice: requestComposerVoice,
    focusComposer,
  };
}
