import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { Share } from "react-native";

import {
  disableSharing,
  enableSharing,
  getShareState,
  shareKeys,
  type ChatShareState,
} from "@/lib/api/share";
import { FRONTEND_URL } from "@/lib/env";

import {
  absoluteShareUrl,
  CREATE_FAILED,
  shareErrorText,
  shareUrlForToken,
  STOP_FAILED,
} from "./helpers";

export function useShareSheet(sessionID: string) {
  const queryClient = useQueryClient();
  // The web's default: share the chat's runs too, unless the owner says no.
  const [includeRuns, setIncludeRuns] = useState(true);
  const [createdUrl, setCreatedUrl] = useState<string | null>(null);
  const state = useQuery({
    queryKey: shareKeys.state(sessionID),
    queryFn: () => getShareState(sessionID),
  });

  function updateState(patch: Partial<ChatShareState>) {
    queryClient.setQueryData<ChatShareState>(shareKeys.state(sessionID), (old) =>
      old ? { ...old, ...patch } : old,
    );
    void queryClient.invalidateQueries({ queryKey: shareKeys.state(sessionID) });
  }

  const create = useMutation({
    mutationFn: () => enableSharing(sessionID, includeRuns),
    onSuccess(share) {
      setCreatedUrl(absoluteShareUrl(share.share_url, FRONTEND_URL));
      updateState({
        is_shared: true,
        share_token: share.share_token,
        auto_share_executions: includeRuns,
      });
    },
  });
  const stop = useMutation({
    mutationFn: () => disableSharing(sessionID),
    onSuccess() {
      setCreatedUrl(null);
      updateState({ is_shared: false, share_token: null });
    },
  });

  const isShared = !!state.data?.is_shared;
  const token = state.data?.share_token ?? null;
  const url = isShared
    ? (createdUrl ?? (token ? shareUrlForToken(token, FRONTEND_URL) : null))
    : null;

  function errorText() {
    if (create.isError) return shareErrorText(create.error, CREATE_FAILED);
    if (stop.isError) return shareErrorText(stop.error, STOP_FAILED);
    return null;
  }

  function createLink() {
    stop.reset();
    create.mutate();
  }

  function stopSharing() {
    create.reset();
    stop.mutate();
  }

  function sendLink() {
    if (url) void Share.share({ message: url });
  }

  function retry() {
    void state.refetch();
  }

  return {
    isLoading: state.isPending,
    isLoadError: !state.data && state.isError,
    retry,
    isShared,
    url,
    // Locked once shared, so the switch always matches what the link shows.
    includeRuns: isShared ? !!state.data?.auto_share_executions : includeRuns,
    setIncludeRuns,
    createLink,
    isCreating: create.isPending,
    stopSharing,
    isStopping: stop.isPending,
    sendLink,
    error: errorText(),
  };
}
