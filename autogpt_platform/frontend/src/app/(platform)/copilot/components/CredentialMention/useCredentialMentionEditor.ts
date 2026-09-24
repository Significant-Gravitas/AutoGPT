import { useLayoutEffect, useRef, useState } from "react";
import { parseCredentialMentions, type CredentialMentionPart } from "./helpers";
import {
  editorSelectionOffset,
  editorTextRuns,
  readMentionEditor,
  setEditorCaret,
} from "./editorHelpers";

interface BadgeMount {
  element: HTMLSpanElement;
  mention: CredentialMentionPart;
}

/** True while the DOM already shows `value` with every credential reference
 *  as a badge. A reference typed or pasted as plain text serializes to the
 *  same value, so the text runs between badges are checked as well. */
function editorShows(editor: HTMLElement, value: string) {
  return (
    readMentionEditor(editor) === value &&
    !editorTextRuns(editor).some((run) =>
      parseCredentialMentions(run).some((part) => typeof part !== "string"),
    )
  );
}

export function useCredentialMentionEditor(
  value: string,
  onMultilineChange?: (multiline: boolean) => void,
) {
  const editorRef = useRef<HTMLDivElement>(null);
  const [badges, setBadges] = useState<BadgeMount[]>([]);
  const composingRef = useRef(false);
  const [compositionCount, setCompositionCount] = useState(0);
  const onMultilineRef = useRef(onMultilineChange);
  onMultilineRef.current = onMultilineChange;

  useLayoutEffect(() => {
    const editor = editorRef.current;
    if (!editor || composingRef.current || editorShows(editor, value)) return;
    const focused = document.activeElement === editor;
    const caret = editorSelectionOffset(editor);
    const mounts: BadgeMount[] = [];
    const nodes = parseCredentialMentions(value).map((part) => {
      if (typeof part === "string") return document.createTextNode(part);
      const element = document.createElement("span");
      element.contentEditable = "false";
      element.dataset.credentialMention = part.token;
      mounts.push({ element, mention: part });
      return element;
    });
    editor.replaceChildren(...nodes);
    setBadges(mounts);
    if (focused) setEditorCaret(editor, Math.min(caret, value.length));
  }, [value, compositionCount]);

  useLayoutEffect(() => {
    if (editorRef.current) onMultilineRef.current?.(true);
  }, [value]);

  function onCompositionStart() {
    composingRef.current = true;
  }

  function onCompositionEnd() {
    composingRef.current = false;
    setCompositionCount((count) => count + 1);
  }

  return { editorRef, badges, onCompositionStart, onCompositionEnd };
}
