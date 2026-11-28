import { Flag, useFlagStatus } from "@/services/feature-flags/use-get-flag";
import { useEffect, useRef, useState, useSyncExternalStore } from "react";
import type { FrameSnapshot } from "./chatFrameHandoff";

// Past this the chat is shown however its load went, so a load that never
// settles cannot leave the frame up for good.
export const FRAME_FALLBACK_MS = 15_000;

/** Whether the server-rendered frame is still the message box: only a page
 *  from the server's HTML has it, until the chat under it can take the box. */
export function useChatFrameHandoff(isChatMounted: boolean) {
  const isHydrating = useSyncExternalStore(
    subscribeToNothing,
    () => false,
    () => true,
  );
  const [startedWithFrame] = useState(isHydrating);
  const [isChatLoaded, setIsChatLoaded] = useState(false);
  const [hasTimedOut, setHasTimedOut] = useState(false);
  const [isHandedOver, setIsHandedOver] = useState(false);
  const snapshot = useRef<FrameSnapshot | null>(null);
  // The chat swaps its whole subtree when this flag answers, so the frame
  // waits for it rather than hand its box to a composer about to be replaced.
  const streamRuntime = useFlagStatus(Flag.COPILOT_STREAM_RUNTIME);
  const isReady =
    isChatMounted && (isChatLoaded || hasTimedOut) && streamRuntime.ready;
  // Once handed over the box stays with the chat, even if the chat reloads.
  if (startedWithFrame && isReady && !isHandedOver) setIsHandedOver(true);
  const isShown = startedWithFrame && !isHandedOver && !isReady;

  useEffect(() => {
    if (!isShown || !isChatMounted) return;
    const timer = setTimeout(() => setHasTimedOut(true), FRAME_FALLBACK_MS);
    return () => clearTimeout(timer);
  }, [isShown, isChatMounted]);

  function keepSnapshot(next: FrameSnapshot | null) {
    snapshot.current = next;
  }

  function takeSnapshot() {
    const taken = snapshot.current;
    snapshot.current = null;
    return taken;
  }

  function reportChatLoaded(isLoaded: boolean) {
    setIsChatLoaded(isLoaded);
  }

  return { isShown, keepSnapshot, takeSnapshot, reportChatLoaded };
}

function subscribeToNothing() {
  return () => {};
}
