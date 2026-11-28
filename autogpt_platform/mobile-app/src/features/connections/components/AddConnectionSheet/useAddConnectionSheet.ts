import { useMutation, useQuery } from "@tanstack/react-query";
import { useState } from "react";

import {
  integrationKeys,
  listCredentials,
  listExpertCredentials,
  listProviders,
} from "@/lib/api/integrations";
import { useMountEffect } from "@/lib/useMountEffect";

import type { PickedCredential } from "../../connectedThisRun";
import { allowErrorText, allowForExpert } from "../../expertAccess";
import { grantableRows, toServices, type ConnectableService } from "../../listHelpers";
import { useConnect } from "../../useConnect";
import { serverTarget, SERVER_ROW, sheetLists } from "./helpers";

// The list of services changes with deploys, not while you look at it.
const PROVIDERS_STALE_MS = 10 * 60_000;

interface Args {
  sessionID: string;
  expertID: string | null;
  onDone: () => void;
  /** Opened for one service: its step starts at once, as a press on its row would. */
  initialService?: ConnectableService | null;
  /** An account was added or allowed. */
  onConnected?: () => void;
}

interface Allow {
  rowKey: string;
  credentialID: string;
}

/** The services anyone can connect; the sheet and its opener share them. */
export function useProviders() {
  return useQuery({
    queryKey: integrationKeys.providers,
    queryFn: listProviders,
    staleTime: PROVIDERS_STALE_MS,
  });
}

export function useAddConnectionSheet({
  sessionID,
  expertID,
  onDone,
  initialService = null,
  onConnected,
}: Args) {
  const [query, setQuery] = useState("");
  const [isAskingServer, setIsAskingServer] = useState(false);
  const [serverText, setServerText] = useState("");
  const isExpert = expertID !== null;

  const providers = useProviders();
  const credentials = useQuery({
    queryKey: integrationKeys.credentials,
    queryFn: listCredentials,
    enabled: isExpert,
  });
  const granted = useQuery({
    queryKey: integrationKeys.expertCredentials(expertID ?? ""),
    queryFn: () => listExpertCredentials(expertID ?? ""),
    enabled: isExpert,
  });
  const allow = useMutation({
    mutationFn: ({ credentialID }: Allow) => allowForExpert(expertID ?? "", [credentialID]),
    onSuccess: finish,
    onError(error, { rowKey }) {
      connect.showError(rowKey, allowErrorText(error));
    },
  });
  const connect = useConnect({
    origin: { sessionID, cardKey: null, expertID },
    onConnected: handleConnected,
  });

  // An account only counts as not yet allowed once both lists are in.
  const accounts =
    credentials.data && granted.data
      ? grantableRows(credentials.data, granted.data.map((item) => item.credentialID))
      : [];
  const lists = sheetLists({ services: toServices(providers.data ?? []), accounts, query });
  const server = serverTarget(serverText);
  const isBusy = connect.busyKey !== null || allow.isPending;

  useMountEffect(() => {
    if (initialService) connectService(initialService);
  });

  function finish() {
    onConnected?.();
    onDone();
  }

  function handleConnected(rowKey: string, credential: PickedCredential) {
    if (!expertID) {
      finish();
      return;
    }
    allow.mutate({ rowKey, credentialID: credential.id });
  }

  function allowAccount(credentialID: string) {
    allow.mutate({ rowKey: `account:${credentialID}`, credentialID });
  }

  function connectService(service: ConnectableService) {
    connect.connectService(service.key, service.name, service.action);
  }

  function askServer() {
    setIsAskingServer(true);
  }

  function cancelServer() {
    setIsAskingServer(false);
  }

  function signInToServer() {
    if (!server || isBusy) return;
    void connect.signIn(SERVER_ROW, { kind: "mcp", ...server });
  }

  function isRowBusy(rowKey: string) {
    return connect.busyKey === rowKey || (allow.isPending && allow.variables?.rowKey === rowKey);
  }

  function retryProviders() {
    void providers.refetch();
  }

  return {
    query,
    setQuery,
    isExpert,
    lists,
    isLoadingServices: providers.isPending,
    hasServicesFailed: !providers.data && providers.isError,
    retryProviders,
    connect,
    isBusy,
    isRowBusy,
    allowAccount,
    connectService,
    isAskingServer,
    askServer,
    cancelServer,
    serverText,
    setServerText,
    canSignInToServer: server !== null,
    signInToServer,
  };
}
