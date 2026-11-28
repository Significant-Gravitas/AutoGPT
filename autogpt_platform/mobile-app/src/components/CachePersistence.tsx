import { identifyUser } from "@/lib/analytics/client";
import { queryPersister, startCachePersistence } from "@/lib/queryPersist";
import { queryClient } from "@/lib/session";
import { useMountEffect } from "@/lib/useMountEffect";

interface Props {
  userID: string;
}

/**
 * Restores and saves the signed-in user's cache, and names them to analytics
 * by id. It renders nothing and mounts only while someone is signed in; when
 * the session ends, the copy in memory and on the phone goes with it.
 */
export function CachePersistence({ userID }: Props) {
  useMountEffect(() => {
    identifyUser(userID);
    const stop = startCachePersistence(queryClient, userID);
    return () => {
      stop();
      queryClient.clear();
      void queryPersister.removeClient();
    };
  });
  return null;
}
