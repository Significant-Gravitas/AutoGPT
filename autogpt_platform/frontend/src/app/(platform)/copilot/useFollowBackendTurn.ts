import { useEffect, useRef } from "react";

interface Args {
  status: string;
  refetchSession: () => Promise<{ data?: unknown }>;
  hasResumedRef: React.MutableRefObject<boolean>;
}

// An answered approval card starts the chat's next turn on the server. The page
// resumes a server-started stream once per mount, after hydration; re-arming
// that resume lets the refetched rows land first, so the replay follows them.
export function useFollowBackendTurn({
  status,
  refetchSession,
  hasResumedRef,
}: Args) {
  const pendingRef = useRef(false);
  const isBusy = status === "streaming" || status === "submitted";
  const followRef = useRef(follow);
  followRef.current = follow;

  function follow() {
    pendingRef.current = false;
    hasResumedRef.current = false;
    refetchSession();
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
