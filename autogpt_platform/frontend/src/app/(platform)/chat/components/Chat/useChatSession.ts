import {
  getGetV2GetSessionQueryKey,
  getGetV2GetSessionQueryOptions,
  postV2CreateSession,
  useGetV2GetSession,
  usePatchV2SessionAssignUser,
  usePostV2CreateSession,
} from "@/app/api/__generated__/endpoints/chat/chat";
import type { SessionDetailResponse } from "@/app/api/__generated__/models/sessionDetailResponse";
import { okData } from "@/app/api/helpers";
import { isValidUUID } from "@/lib/utils";
import { Key, storage } from "@/services/storage/local-storage";
import { useQueryClient } from "@tanstack/react-query";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { toast } from "sonner";

interface UseChatSessionArgs {
  urlSessionId?: string | null;
  autoCreate?: boolean;
}

export function useChatSession({
  urlSessionId,
  autoCreate = false,
}: UseChatSessionArgs = {}) {
  const queryClient = useQueryClient();
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const justCreatedSessionIdRef = useRef<string | null>(null);

  useEffect(() => {
    if (urlSessionId) {
      if (!isValidUUID(urlSessionId)) {
        console.error("Invalid session ID format:", urlSessionId);
        toast.error("Invalid session ID", {
          description:
            "The session ID in the URL is not valid. Starting a new session...",
        });
        setSessionId(null);
        storage.clean(Key.CHAT_SESSION_ID);
        return;
      }
      setSessionId(urlSessionId);
      storage.set(Key.CHAT_SESSION_ID, urlSessionId);
    } else {
      const storedSessionId = storage.get(Key.CHAT_SESSION_ID);
      if (storedSessionId) {
        if (!isValidUUID(storedSessionId)) {
          console.error("Invalid stored session ID:", storedSessionId);
          storage.clean(Key.CHAT_SESSION_ID);
          setSessionId(null);
        } else {
          setSessionId(storedSessionId);
        }
      } else if (autoCreate) {
        setSessionId(null);
      }
    }
  }, [urlSessionId, autoCreate]);

  const {
    mutateAsync: createSessionMutation,
    isPending: isCreating,
    error: createError,
  } = usePostV2CreateSession();

  const {
    data: sessionData,
    isLoading: isLoadingSession,
    error: loadError,
    refetch,
  } = useGetV2GetSession(sessionId || "", {
    query: {
      enabled: !!sessionId,
      select: okData,
      staleTime: Infinity, // Never mark as stale
      refetchOnMount: false, // Don't refetch on component mount
      refetchOnWindowFocus: false, // Don't refetch when window regains focus
      refetchOnReconnect: false, // Don't refetch when network reconnects
      retry: 1,
    },
  });

  const { mutateAsync: claimSessionMutation } = usePatchV2SessionAssignUser();

  const session = useMemo(() => {
    if (sessionData) return sessionData;

    if (sessionId && justCreatedSessionIdRef.current === sessionId) {
      return {
        id: sessionId,
        user_id: null,
        messages: [],
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      } as SessionDetailResponse;
    }
    return null;
  }, [sessionData, sessionId]);

  const messages = session?.messages || [];
  const isLoading = isCreating || isLoadingSession;

  useEffect(() => {
    if (createError) {
      setError(
        createError instanceof Error
          ? createError
          : new Error("Failed to create session"),
      );
    } else if (loadError) {
      setError(
        loadError instanceof Error
          ? loadError
          : new Error("Failed to load session"),
      );
    } else {
      setError(null);
    }
  }, [createError, loadError]);

  const createSession = useCallback(
    async function createSession() {
      try {
        setError(null);
        const response = await postV2CreateSession({
          body: JSON.stringify({}),
        });
        if (response.status !== 200) {
          throw new Error("Failed to create session");
        }
        const newSessionId = response.data.id;
        setSessionId(newSessionId);
        storage.set(Key.CHAT_SESSION_ID, newSessionId);
        justCreatedSessionIdRef.current = newSessionId;
        setTimeout(() => {
          if (justCreatedSessionIdRef.current === newSessionId) {
            justCreatedSessionIdRef.current = null;
          }
        }, 10000);
        return newSessionId;
      } catch (err) {
        const error =
          err instanceof Error ? err : new Error("Failed to create session");
        setError(error);
        toast.error("Failed to create chat session", {
          description: error.message,
        });
        throw error;
      }
    },
    [createSessionMutation],
  );

  const loadSession = useCallback(
    async function loadSession(id: string) {
      try {
        setError(null);
        // Invalidate the query cache for this session to force a fresh fetch
        await queryClient.invalidateQueries({
          queryKey: getGetV2GetSessionQueryKey(id),
        });
        // Set sessionId after invalidation to ensure the hook refetches
        setSessionId(id);
        storage.set(Key.CHAT_SESSION_ID, id);
        // Force fetch with fresh data (bypass cache)
        const queryOptions = getGetV2GetSessionQueryOptions(id, {
          query: {
            staleTime: 0, // Force fresh fetch
            retry: 1,
          },
        });
        const result = await queryClient.fetchQuery(queryOptions);
        if (!result || ("status" in result && result.status !== 200)) {
          console.warn("Session not found on server, clearing local state");
          storage.clean(Key.CHAT_SESSION_ID);
          setSessionId(null);
          throw new Error("Session not found");
        }
      } catch (err) {
        const error =
          err instanceof Error ? err : new Error("Failed to load session");
        setError(error);
        throw error;
      }
    },
    [queryClient],
  );

  const refreshSession = useCallback(
    async function refreshSession() {
      if (!sessionId) {
        console.log("[refreshSession] Skipping - no session ID");
        return;
      }
      try {
        setError(null);
        await refetch();
      } catch (err) {
        const error =
          err instanceof Error ? err : new Error("Failed to refresh session");
        setError(error);
        throw error;
      }
    },
    [sessionId, refetch],
  );

  const claimSession = useCallback(
    async function claimSession(id: string) {
      try {
        setError(null);
        await claimSessionMutation({ sessionId: id });
        if (justCreatedSessionIdRef.current === id) {
          justCreatedSessionIdRef.current = null;
        }
        await queryClient.invalidateQueries({
          queryKey: getGetV2GetSessionQueryKey(id),
        });
        await refetch();
        toast.success("Session claimed successfully", {
          description: "Your chat history has been saved to your account",
        });
      } catch (err: unknown) {
        const error =
          err instanceof Error ? err : new Error("Failed to claim session");
        const is404 =
          (typeof err === "object" &&
            err !== null &&
            "status" in err &&
            err.status === 404) ||
          (typeof err === "object" &&
            err !== null &&
            "response" in err &&
            typeof err.response === "object" &&
            err.response !== null &&
            "status" in err.response &&
            err.response.status === 404);
        if (!is404) {
          setError(error);
          toast.error("Failed to claim session", {
            description: error.message || "Unable to claim session",
          });
        }
        throw error;
      }
    },
    [claimSessionMutation, queryClient, refetch],
  );

  const clearSession = useCallback(function clearSession() {
    setSessionId(null);
    setError(null);
    storage.clean(Key.CHAT_SESSION_ID);
    justCreatedSessionIdRef.current = null;
  }, []);

  return {
    session,
    sessionId,
    messages,
    isLoading,
    isCreating,
    error,
    createSession,
    loadSession,
    refreshSession,
    claimSession,
    clearSession,
  };
}
