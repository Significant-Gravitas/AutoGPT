import { useEffect, useRef } from "react";
import { hasActiveBackendStream } from "./helpers";

interface Args {
  status: string;
  refetchSession: () => Promise<{ data?: unknown }>;
  resumeStreamRef: React.MutableRefObject<() => void>;
}

// An answered approval card starts the chat's next turn on the server. The page
// resumes a server-started stream only once per mount, so it is followed here:
// at once when the chat is idle, or when the running turn ends.
export function useFollowBackendTurn({
  status,
  refetchSession,
  resumeStreamRef,
}: Args) {
  const pendingRef = useRef(false);
  const isBusy = status === "streaming" || status === "submitted";

  const followRef = useRef(follow);
  followRef.current = follow;

  async function follow() {
    pendingRef.current = false;
    const result = await refetchSession();
    if (hasActiveBackendStream(result)) resumeStreamRef.current();
  }

  useEffect(() => {
    if (!isBusy && pendingRef.current) followRef.current();
  }, [isBusy]);

  function followBackendTurn() {
    pendingRef.current = true;
    if (!isBusy) follow();
  }

  return { followBackendTurn };
}
