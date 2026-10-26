import {
  getListExpertSetupItemsQueryKey,
  useGrantExpertCredentials,
  useListExpertSetupItems,
  useListExperts,
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
import { useEffect, useState } from "react";
import { buildTeam, chooseAsk, type ConnectAsk, joinNames } from "./helpers";

const COMPOSER_INPUT_ID = "chat-input-empty";
const SETUP_RETRIES = 3;

export function useIntroStandup({ path }: { path: string }) {
  const queryClient = useQueryClient();
  const { toast } = useToast();
  const [connectDismissed, setConnectDismissed] = useState(false);
  // The ask the open dialog was opened for; a re-read can move the live one.
  const [connecting, setConnecting] = useState<ConnectAsk | null>(null);

  const hiresQuery = useListExperts({
    query: {
      select: (res) => getHiredExperts((okData(res) ?? []) as Expert[]),
      refetchInterval: (query) =>
        isSettingUp((okData(query.state.data) ?? []) as Expert[])
          ? SETUP_POLL_MS
          : false,
    },
  });
  const setupQuery = useListExpertSetupItems({
    query: {
      select: (res) => okData(res) ?? [],
      // The re-read after an install is the only way to learn that hire's
      // gaps, so a flaky one gets a few tries before it is given up on.
      retry: SETUP_RETRIES,
      retryDelay: (attempt) => Math.min(500 * 2 ** attempt, 4_000),
    },
  });
  const providersQuery = useGetV1ListProviders({
    query: { select: (res) => (res.status === 200 ? res.data : []) },
  });
  const connectable = new Set(
    toConnectableProviders(providersQuery.data ?? []).map((p) => p.id),
  );
  const { mutateAsync: grant } = useGrantExpertCredentials();

  const unchecked = useUncheckedHires({
    hires: hiresQuery.data,
    hiresUpdatedAt: hiresQuery.dataUpdatedAt,
    setupUpdatedAt: setupQuery.dataUpdatedAt,
    setupFailed: setupQuery.isError && !setupQuery.isFetching,
    refetchSetup: setupQuery.refetch,
  });
  const setupItems = setupQuery.data ?? [];
  // No status is shown before the setup gaps have been read, so none reads
  // Ready only to change; a read that failed shows as unchecked instead.
  const team = setupQuery.isFetched
    ? buildTeam({
        hires: hiresQuery.data ?? [],
        setupItems: setupQuery.data ?? null,
        unchecked,
      })
    : [];
  // Nothing is asked until hires, setup gaps and providers have been read,
  // and nothing once a finished install's setup gaps could not be.
  const isSettled =
    hiresQuery.data !== undefined &&
    setupQuery.data !== undefined &&
    providersQuery.data !== undefined &&
    !unchecked.failed;
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
    const target = connecting;
    setConnecting(null);
    if (!target) return;
    const results = await Promise.allSettled(
      target.expertIds.map((expertId) =>
        grant({ expertId, data: { credential_ids: [credential.id] } }).then(
          () => invalidateExpertGrantQueries(queryClient, expertId),
        ),
      ),
    );
    const failed = target.names.filter(
      (_, index) => results[index].status === "rejected",
    );
    if (failed.length > 0) {
      toast({
        title: `Couldn't give ${joinNames(failed)} access to ${target.service}`,
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
    connecting,
    openConnect: () => setConnecting(ask?.kind === "connect" ? ask : null),
    closeConnect: () => setConnecting(null),
    handleConnected,
    dismissConnect: () => setConnectDismissed(true),
    startVoice: requestComposerVoice,
    focusComposer,
    // The same check the composer's mic makes; the card renders after mount.
    canRecord:
      typeof navigator !== "undefined" &&
      !!navigator.mediaDevices?.getUserMedia,
  };
}

// A hire's schedules, and so its setup gaps, exist only once its install has
// finished. The hires that finished after the last setup read are unchecked
// until a re-read lands; `failed` once every retry of it has.
function useUncheckedHires(args: {
  hires: Expert[] | undefined;
  hiresUpdatedAt: number;
  setupUpdatedAt: number;
  setupFailed: boolean;
  refetchSetup: () => unknown;
}) {
  const installing = (args.hires ?? [])
    .filter((expert) => expert.setup_status === "installing")
    .map((expert) => expert.id);
  // A failed install created no schedules, so there is nothing to re-read.
  const failedInstalls = (args.hires ?? [])
    .filter((expert) => expert.setup_status === "failed")
    .map((expert) => expert.id);
  const installingKey = installing.join(",");
  const [seenKey, setSeenKey] = useState<string | null>(null);
  const [finished, setFinished] = useState({ ids: [] as string[], at: 0 });

  let current = finished;
  if (args.hires && installingKey !== seenKey) {
    const left = (seenKey ?? "")
      .split(",")
      .filter(
        (id) => id && !installing.includes(id) && !failedInstalls.includes(id),
      );
    setSeenKey(installingKey);
    if (left.length > 0) {
      // Hires a re-read has already covered stay checked.
      const pending = args.setupUpdatedAt >= finished.at ? [] : finished.ids;
      current = { ids: [...pending, ...left], at: args.hiresUpdatedAt };
      setFinished(current);
    }
  }

  const queryClient = useQueryClient();
  const { refetchSetup } = args;
  useEffect(() => {
    if (finished.at === 0) return;
    // A read still on its way was sent before the install finished, and with
    // no data yet a refetch joins it instead of asking again.
    void queryClient
      .cancelQueries({ queryKey: getListExpertSetupItemsQueryKey() })
      .then(() => refetchSetup());
  }, [finished.at, queryClient, refetchSetup]);

  const ids = args.setupUpdatedAt >= current.at ? [] : current.ids;
  return { ids, failed: ids.length > 0 && args.setupFailed };
}
