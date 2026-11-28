import { isTouchDevice } from "@/lib/platform";
import { useContext, useLayoutEffect, useRef } from "react";
import { setEditorCaret } from "../CredentialMention/editorHelpers";
import {
  ChatFrameHandoffContext,
  mergeHandoffText,
  type FrameSnapshot,
} from "../ChatFrame/chatFrameHandoff";
import { focusComposer } from "./helpers";

interface Args {
  inputId: string;
  draftKey?: string;
  /** This composer is the page's own, the one the chat frame hands to. */
  takesFrame: boolean;
  value: string;
  setValue: (value: string) => void;
  autoFocus: boolean;
}

interface Restore {
  value: string;
  start: number;
  end: number;
  direction: FrameSnapshot["selectionDirection"];
  focus: boolean;
  send: boolean;
  /** Set while the composer's value has yet to catch up with `value`. */
  awaitsValue: boolean;
}

// Where a focused composer's caret sat when it unmounted, for the copy of it
// that mounts in its place (a flag resolving swaps the chat's whole subtree).
let kept: { key: string; at: number; restore: Restore } | null = null;
const KEPT_FOR_MS = 1000;

/** Carries the text, caret, selection, focus and any pending send from the
 *  chat frame's message box into this one, and keeps the caret where it was
 *  when this composer is unmounted and mounted again. */
export function useComposerHandoff({
  inputId,
  draftKey,
  takesFrame,
  value,
  setValue,
  autoFocus,
}: Args) {
  const { isFrameShown, takeSnapshot } = useContext(ChatFrameHandoffContext);
  const pending = useRef<Restore | null>(null);
  const keepKey = `${inputId}\u0000${draftKey ?? ""}`;

  // The frame records its box as it unmounts, in the same commit that shows
  // this composer, so the snapshot is there by the time this runs.
  useLayoutEffect(() => {
    if (!takesFrame || isFrameShown) return;
    const snapshot = takeSnapshot();
    if (!snapshot) return;
    const merged = mergeHandoffText(value, snapshot.value);
    const start = merged.typedEmpty
      ? merged.value.length
      : snapshot.selectionStart + merged.offset;
    const end = merged.typedEmpty
      ? merged.value.length
      : snapshot.selectionEnd + merged.offset;
    pending.current = {
      value: merged.value,
      start,
      end,
      direction: snapshot.selectionDirection,
      focus:
        snapshot.focused ||
        (autoFocus &&
          !isTouchDevice() &&
          document.activeElement === document.body),
      send: snapshot.sendPending,
      awaitsValue: merged.value !== value,
    };
    if (merged.value !== value) setValue(merged.value);
  }, [takesFrame, isFrameShown, takeSnapshot, value, setValue, autoFocus]);

  useLayoutEffect(() => {
    if (kept?.key === keepKey && performance.now() - kept.at < KEPT_FOR_MS) {
      pending.current = kept.restore;
    }
    kept = null;
    return () => {
      const input = document.getElementById(inputId);
      if (!(input instanceof HTMLTextAreaElement)) return;
      if (input !== document.activeElement) return;
      kept = {
        key: keepKey,
        at: performance.now(),
        restore: {
          value: input.value,
          start: input.selectionStart,
          end: input.selectionEnd,
          direction: input.selectionDirection ?? "none",
          focus: true,
          send: false,
          awaitsValue: false,
        },
      };
    };
  }, [keepKey, inputId]);

  useLayoutEffect(() => {
    const restore = pending.current;
    if (!restore) return;
    if (restore.value !== value) {
      if (restore.awaitsValue) restore.awaitsValue = false;
      else pending.current = null;
      return;
    }
    const input = document.getElementById(inputId);
    // A box that is disabled for the moment (the chat reconnecting) takes the
    // focus and any pending send once it is enabled again.
    if (input && isDisabled(input) && (restore.focus || restore.send)) return;
    pending.current = null;
    applyRestore(inputId, restore);
  });
}

function isDisabled(input: HTMLElement) {
  return input instanceof HTMLTextAreaElement
    ? input.disabled
    : input.getAttribute("aria-disabled") === "true";
}

function applyRestore(inputId: string, restore: Restore) {
  const input = document.getElementById(inputId);
  if (!input) return;
  if (restore.focus) focusComposer(inputId);
  if (input instanceof HTMLTextAreaElement) {
    input.setSelectionRange(restore.start, restore.end, restore.direction);
  } else if (document.activeElement === input) {
    setEditorCaret(input, restore.start, restore.end);
  }
  if (restore.send) input.closest("form")?.requestSubmit();
}
