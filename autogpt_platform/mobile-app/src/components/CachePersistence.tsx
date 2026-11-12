import { queryPersister, startCachePersistence } from "@/lib/queryPersist";
import { queryClient } from "@/lib/session";
import { useMountEffect } from "@/lib/useMountEffect";

interface Props {
  userID: string;
}

/**
 * Restores and saves the signed-in user's cache. It renders nothing and
 * mounts only while someone is signed in; when the session ends, the copy in
 * memory and on the phone goes with it.
 */
export function CachePersistence({ userID }: Props) {
  useMountEffect(() => {
    const stop = startCachePersistence(queryClient, userID);
    return () => {
      stop();
      queryClient.clear();
      void queryPersister.removeClient();
    };
  });
  return null;
}
