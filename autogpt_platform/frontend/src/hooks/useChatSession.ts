import { useCallback, useEffect, useState, useRef, useMemo } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import {
  usePostV2CreateSession,
  postV2CreateSession,
  useGetV2GetSession,
  usePatchV2SessionAssignUser,
  getGetV2GetSessionQueryKey,
} from "@/app/api/__generated__/endpoints/chat/chat";
import type { SessionDetailResponse } from "@/app/api/__generated__/models/sessionDetailResponse";
import { storage, Key } from "@/services/storage/local-storage";
import { isValidUUID } from "@/lib/utils";

interface UseChatSessionArgs {
  urlSessionId?: string | null;
  autoCreate?: boolean;
}

interface UseChatSessionResult {
  session: SessionDetailResponse | null;
  sessionId: string | null; // Direct access to session ID state
  messages: SessionDetailResponse["messages"];
  isLoading: boolean;
  isCreating: boolean;
  error: Error | null;
  createSession: () => Promise<string>; // Return session ID
  loadSession: (sessionId: string) => Promise<void>;
  refreshSession: () => Promise<void>;
  claimSession: (sessionId: string) => Promise<void>;
  clearSession: () => void;
}

export function useChatSession({
  urlSessionId,
  autoCreate = false,
}: UseChatSessionArgs = {}): UseChatSessionResult {
  const queryClient = useQueryClient();
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const justCreatedSessionIdRef = useRef<string | null>(null);

  // Initialize session ID from URL or localStorage
  useEffect(
    function initializeSessionId() {
      if (urlSessionId) {
        // Validate UUID format
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
          // Validate stored session ID as well
          if (!isValidUUID(storedSessionId)) {
            console.error("Invalid stored session ID:", storedSessionId);
            storage.clean(Key.CHAT_SESSION_ID);
            setSessionId(null);
          } else {
            setSessionId(storedSessionId);
          }
        } else if (autoCreate) {
          // Auto-create will be handled by the mutation below
          setSessionId(null);
        }
      }
    },
    [urlSessionId, autoCreate],
  );

  // Create session mutation
  const {
    mutateAsync: createSessionMutation,
    isPending: isCreating,
    error: createError,
  } = usePostV2CreateSession();

  // Get session query - runs for any valid session (URL or locally created)
  const {
    data: sessionData,
    isLoading: isLoadingSession,
    error: loadError,
    refetch,
  } = useGetV2GetSession(sessionId || "", {
    query: {
      enabled: !!sessionId, // Fetch whenever we have a session ID
      staleTime: 30000, // Consider data fresh for 30 seconds
      retry: 1,
      // Error handling is done in useChatPage via the error state
    },
  });

  // Claim session mutation (assign user to anonymous session)
  const { mutateAsync: claimSessionMutation } = usePatchV2SessionAssignUser();

  // Extract session from response with type guard
  // Once we have session data from the backend, use it
  // While waiting for the first fetch, create a minimal object for just-created sessions
  const session: SessionDetailResponse | null = useMemo(() => {
    // If we have real session data from GET query, use it
    if (sessionData?.status === 200) {
      return sessionData.data;
    }

    // If we just created a session and are waiting for the first fetch, create a minimal object
    // This prevents a blank page while the GET query loads
    if (sessionId && justCreatedSessionIdRef.current === sessionId) {
      return {
        id: sessionId,
        user_id: null, // Placeholder - actual value set by backend during creation
        messages: [],
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      } as SessionDetailResponse;
    }

    return null;
  }, [sessionData, sessionId]);

  const messages = session?.messages || [];

  // Combined loading state
  const isLoading = isCreating || isLoadingSession;

  // Combined error state
  useEffect(
    function updateError() {
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
    },
    [createError, loadError],
  );

  const createSession = useCallback(
    async function createSession(): Promise<string> {
      try {
        setError(null);
        // Call the API function directly with empty body to satisfy Content-Type: application/json
        const response = await postV2CreateSession({
          body: JSON.stringify({}),
        });

        // Type guard to ensure we have a successful response
        if (response.status !== 200) {
          throw new Error("Failed to create session");
        }

        const newSessionId = response.data.id;

        setSessionId(newSessionId);
        storage.set(Key.CHAT_SESSION_ID, newSessionId);

        // Mark this session as "just created" so we can create a minimal object for it
        justCreatedSessionIdRef.current = newSessionId;

        // Clear the "just created" flag after 10 seconds
        // By then, the session should have been claimed or the user should have started using it
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
        setSessionId(id);
        storage.set(Key.CHAT_SESSION_ID, id);

        // Attempt to fetch the session to verify it exists
        const result = await refetch();

        // If session doesn't exist (404), clear it and throw error
        if (!result.data || result.isError) {
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
    [refetch],
  );

  const refreshSession = useCallback(
    async function refreshSession() {
      // Refresh session data from backend (works for all sessions now)
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

        // Session was successfully claimed, so we know it exists on the server
        // Clear the "just created" flag for this session
        if (justCreatedSessionIdRef.current === id) {
          justCreatedSessionIdRef.current = null;
        }

        // Invalidate and refetch the session query to get updated user_id
        await queryClient.invalidateQueries({
          queryKey: getGetV2GetSessionQueryKey(id),
        });

        // Force a refetch to sync the session data
        await refetch();

        toast.success("Session claimed successfully", {
          description: "Your chat history has been saved to your account",
        });
      } catch (err: unknown) {
        const error =
          err instanceof Error ? err : new Error("Failed to claim session");

        // Check if this is a 404 error (API errors may have status or response.status)
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

        // Don't show toast for 404 - it will be handled by the caller
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
    sessionId, // Return direct access to sessionId state
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
