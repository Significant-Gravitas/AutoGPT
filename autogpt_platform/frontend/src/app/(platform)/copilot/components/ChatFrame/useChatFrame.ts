import { isTouchDevice } from "@/lib/platform";
import {
  useInsertionEffect,
  useLayoutEffect,
  useRef,
  useSyncExternalStore,
  type FormEvent,
} from "react";
import { getInputPlaceholder } from "../EmptySession/helpers";
import {
  markSendPending,
  readFrame,
  type FrameSnapshot,
} from "./chatFrameHandoff";
import { FRAME_HYDRATED_ATTRIBUTE } from "./frameScript";

export function useChatFrame(
  hasSession: boolean,
  onUnmount?: (snapshot: FrameSnapshot | null) => void,
) {
  const ref = useRef<HTMLDivElement>(null);
  const onUnmountRef = useRef(onUnmount);
  useInsertionEffect(() => {
    onUnmountRef.current = onUnmount;
  });
  const isHydrating = useSyncExternalStore(
    subscribeToNothing,
    () => false,
    () => true,
  );

  // The cleanup records the box for the chat: it runs before React removes
  // the node, so the box is still there and still focused.
  useLayoutEffect(() => {
    const root = ref.current;
    if (!root) return;
    root.setAttribute(FRAME_HYDRATED_ATTRIBUTE, "");
    const input = root.querySelector("textarea");
    if (input && !hasSession)
      input.placeholder = getInputPlaceholder(window.innerWidth);
    if (input && !isTouchDevice() && document.activeElement === document.body)
      input.focus({ preventScroll: true });
    return () => onUnmountRef.current?.(readFrame(root));
  }, [hasSession]);

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const text = new FormData(event.currentTarget).get("message");
    if (ref.current && typeof text === "string" && text.trim())
      markSendPending(ref.current);
  }

  return {
    ref,
    isHydrating,
    composer: {
      stacked: !hasSession,
      placeholder: hasSession
        ? "What else can I help with?"
        : getInputPlaceholder(),
      onSubmit: handleSubmit,
    },
  };
}

function subscribeToNothing() {
  return () => {};
}
