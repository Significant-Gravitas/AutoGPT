import { createContext, useContext, useEffect } from "react";

/** What the frame's message box held when the real composer took over. */
export interface FrameSnapshot {
  value: string;
  selectionStart: number;
  selectionEnd: number;
  selectionDirection: "forward" | "backward" | "none";
  focused: boolean;
  /** Enter or send was pressed before the chat could take the message. */
  sendPending: boolean;
}

interface ChatFrameHandoff {
  isFrameShown: boolean;
  /** The frame's last state, once; null after the first call. */
  takeSnapshot: () => FrameSnapshot | null;
  reportChatLoaded: (isLoaded: boolean) => void;
}

export const ChatFrameHandoffContext = createContext<ChatFrameHandoff>({
  isFrameShown: false,
  takeSnapshot: () => null,
  reportChatLoaded() {},
});

export const SEND_PENDING_ATTRIBUTE = "data-send-pending";
export const FRAME_LOADING_TEXT = "Loading your chat…";
export const FRAME_PENDING_TEXT =
  "Your message will send as soon as your chat has loaded";

/** Marks the frame's send as waiting for the chat. The frame's inline script
 *  does the same before React has loaded. */
export function markSendPending(root: HTMLElement) {
  root.setAttribute(SEND_PENDING_ATTRIBUTE, "");
  const status = root.querySelector('[role="status"]');
  if (status) status.textContent = FRAME_PENDING_TEXT;
}

/** Tells the page whether the chat can take the message box right now, so the
 *  frame hands it over only while it can. `null` leaves the report to another
 *  part of the chat. */
export function useReportChatLoaded(isLoaded: boolean | null) {
  const { reportChatLoaded } = useContext(ChatFrameHandoffContext);
  useEffect(() => {
    if (isLoaded !== null) reportChatLoaded(isLoaded);
  }, [isLoaded, reportChatLoaded]);
}

export function readFrame(root: HTMLElement): FrameSnapshot | null {
  const input = root.querySelector("textarea");
  if (!input) return null;
  return {
    value: input.value,
    selectionStart: input.selectionStart,
    selectionEnd: input.selectionEnd,
    selectionDirection: input.selectionDirection ?? "none",
    focused: document.activeElement === input,
    sendPending:
      root.hasAttribute(SEND_PENDING_ATTRIBUTE) && input.value.trim() !== "",
  };
}

/** The composer's text once the frame's is handed to it, and where the frame's
 *  caret lands in it. A draft the composer already held stays in front, so
 *  neither is lost. */
export function mergeHandoffText(draft: string, typed: string) {
  if (!typed) return { value: draft, offset: draft.length, typedEmpty: true };
  if (!draft.trim() || draft === typed)
    return { value: typed, offset: 0, typedEmpty: false };
  const prefix = `${draft}\n\n`;
  return {
    value: `${prefix}${typed}`,
    offset: prefix.length,
    typedEmpty: false,
  };
}
