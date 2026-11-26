import { useMutation, useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { Alert } from "react-native";

import {
  integrationKeys,
  listExpertCredentials,
  listExpertSetupItems,
  listProviders,
} from "@/lib/api/integrations";

import { allowErrorText, allowForExpert, removeFromExpert } from "../../expertAccess";
import {
  expertNeeds,
  grantedRows,
  signInServerURL,
  toServices,
  type AccountRow,
} from "../../listHelpers";
import { useConnect } from "../../useConnect";
import { needRows, type NeedRow } from "./helpers";

const PROVIDERS_STALE_MS = 10 * 60_000;

interface Args {
  sessionID: string;
  expertID: string;
  expertName: string;
  isOpen: boolean;
}

interface Allow {
  rowKey: string;
  credentialID: string;
}

function grantedKey(credentialID: string) {
  return `granted:${credentialID}`;
}

export function useExpertConnections({ sessionID, expertID, expertName, isOpen }: Args) {
  const [isAdding, setIsAdding] = useState(false);
  // Fetched while the panel is open; each opening refreshes stale lists.
  const granted = useQuery({
    queryKey: integrationKeys.expertCredentials(expertID),
    queryFn: () => listExpertCredentials(expertID),
    enabled: isOpen,
  });
  const setup = useQuery({
    queryKey: integrationKeys.expertSetup,
    queryFn: listExpertSetupItems,
    enabled: isOpen,
  });
  const providers = useQuery({
    queryKey: integrationKeys.providers,
    queryFn: listProviders,
    enabled: isOpen,
    staleTime: PROVIDERS_STALE_MS,
  });
  const allow = useMutation({
    mutationFn: ({ credentialID }: Allow) => allowForExpert(expertID, [credentialID]),
    onError(error, { rowKey }) {
      connect.showError(rowKey, allowErrorText(error));
    },
  });
  const remove = useMutation({
    mutationFn: (credentialID: string) => removeFromExpert(expertID, credentialID),
    onError(_error, credentialID) {
      connect.showError(grantedKey(credentialID), "Couldn't remove access. Try again.");
    },
  });
  const connect = useConnect({
    origin: { sessionID, cardKey: null, expertID },
    onConnected: (rowKey, credential) => allow.mutate({ rowKey, credentialID: credential.id }),
  });

  const grantedIDs = granted.data?.map((item) => item.credentialID) ?? [];
  const accounts = grantedRows(granted.data ?? []);
  const needs = needRows({
    needs: expertNeeds(setup.data ?? [], expertID),
    services: providers.data ? toServices(providers.data) : providers.isError ? [] : null,
    grantedIDs,
  }).map((need) => {
    const action = need.action?.kind === "connect" ? need.action.service.action : null;
    return { ...need, usesToken: connect.usesToken(need.key, signInServerURL(action)) };
  });
  const isBusy = connect.busyKey !== null || allow.isPending || remove.isPending;

  function confirmRemove(account: AccountRow) {
    Alert.alert(
      `Remove ${account.name}?`,
      `${expertName} won't be able to use this account any more. It stays connected, and you can allow it again later.`,
      [
        { text: "Cancel", style: "cancel" },
        { text: "Remove", style: "destructive", onPress: () => remove.mutate(account.id) },
      ],
    );
  }

  function handleNeed(row: NeedRow) {
    if (!row.action) return;
    if (row.action.kind === "allow") {
      allow.mutate({ rowKey: row.key, credentialID: row.action.credentialID });
      return;
    }
    const { service } = row.action;
    connect.connectService(service.key, service.name, service.action);
  }

  function isNeedBusy(row: NeedRow) {
    return connect.busyKey === row.key || (allow.isPending && allow.variables?.rowKey === row.key);
  }

  function isRemoving(account: AccountRow) {
    return remove.isPending && remove.variables === account.id;
  }

  function openAdd() {
    setIsAdding(true);
  }

  function closeAdd() {
    setIsAdding(false);
  }

  function retry() {
    void granted.refetch();
    void setup.refetch();
  }

  return {
    accounts,
    needs,
    isLoading: !granted.data && granted.isPending,
    // A failed needs list would otherwise read as "No connections yet".
    hasFailed: (!granted.data && granted.isError) || (!setup.data && setup.isError),
    // Empty only once the needs are known too, so the empty state never flashes.
    isEmpty: !!granted.data && accounts.length === 0 && needs.length === 0 && !setup.isPending,
    retry,
    connect,
    isBusy,
    confirmRemove,
    isRemoving,
    removeError: (account: AccountRow) => connect.errors[grantedKey(account.id)] ?? null,
    handleNeed,
    isNeedBusy,
    isAdding,
    openAdd,
    closeAdd,
  };
}
