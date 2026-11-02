import { router } from "expo-router";
import { useEffect, useState, useSyncExternalStore } from "react";

import { useSignedIn } from "../auth/SessionProvider";
import { ConversationController } from "./conversationController";

/**
 * One controller per conversation screen. A draft (`sessionID` null) gets
 * its server session on the first send, and the route adopts the new id
 * without remounting the screen.
 */
export function useConversation(
  sessionID: string | null,
  expertID: string | null,
) {
  const { api } = useSignedIn();
  const [controller] = useState(
    () =>
      new ConversationController(api, {
        sessionID,
        expertID,
        onSessionCreated: (id) => router.setParams({ sessionId: id }),
      }),
  );

  useEffect(() => {
    controller.start();
    return () => controller.dispose();
  }, [controller]);

  const snapshot = useSyncExternalStore(
    controller.subscribe,
    controller.getSnapshot,
  );
  return { snapshot, controller };
}
