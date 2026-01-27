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
import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useRef, useState } from "react";
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
        return;
      }
      setSessionId(urlSessionId);
    } else if (autoCreate) {
      setSessionId(null);
    } else {
      setSessionId(null);
    }
  }, [urlSessionId, autoCreate]);

  const { isPending: isCreating, error: createError } =
    usePostV2CreateSession();

  const {
    data: sessionData,
    isLoading: isLoadingSession,
    error: loadError,
    refetch,
  } = useGetV2GetSession(sessionId || "", {
    query: {
      enabled: !!sessionId,
      select: okData,
      staleTime: 0,
      retry: shouldRetrySessionLoad,
      retryDelay: getSessionRetryDelay,
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
  }

  async function loadSession(id: string) {
    try {
      setError(null);
      // Invalidate the query cache for this session to force a fresh fetch
      await queryClient.invalidateQueries({
        queryKey: getGetV2GetSessionQueryKey(id),
      });
      // Set sessionId after invalidation to ensure the hook refetches
      setSessionId(id);
      // Force fetch with fresh data (bypass cache)
      const queryOptions = getGetV2GetSessionQueryOptions(id, {
        query: {
          staleTime: 0, // Force fresh fetch
          retry: shouldRetrySessionLoad,
          retryDelay: getSessionRetryDelay,
        },
      });
      const result = await queryClient.fetchQuery(queryOptions);
      if (!result || ("status" in result && result.status !== 200)) {
        console.warn("Session not found on server");
        setSessionId(null);
        throw new Error("Session not found");
      }
    } catch (err) {
      const error =
        err instanceof Error ? err : new Error("Failed to load session");
      setError(error);
      throw error;
    }
  }

  async function refreshSession() {
    if (!sessionId) return;
    try {
      setError(null);
      await refetch();
    } catch (err) {
      const error =
        err instanceof Error ? err : new Error("Failed to refresh session");
      setError(error);
      throw error;
    }
  }

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
  }

  function clearSession() {
    setSessionId(null);
    setError(null);
    justCreatedSessionIdRef.current = null;
  }

  return {
    session,
    sessionId,
    messages,
    isLoading,
    isCreating,
    error,
    isSessionNotFound: isNotFoundError(loadError),
    createSession,
    loadSession,
    refreshSession,
    claimSession,
    clearSession,
  };
}

function isNotFoundError(error: unknown): boolean {
  if (!error || typeof error !== "object") return false;
  if ("status" in error && error.status === 404) return true;
  if (
    "response" in error &&
    typeof error.response === "object" &&
    error.response !== null &&
    "status" in error.response &&
    error.response.status === 404
  ) {
    return true;
  }
  return false;
}

function shouldRetrySessionLoad(failureCount: number, error: unknown): boolean {
  if (!isNotFoundError(error)) return false;
  return failureCount <= 2;
}

function getSessionRetryDelay(attemptIndex: number): number {
  if (attemptIndex === 0) return 3000;
  if (attemptIndex === 1) return 5000;
  return 0;
}
