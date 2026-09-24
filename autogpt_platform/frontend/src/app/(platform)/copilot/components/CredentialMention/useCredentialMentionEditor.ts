import { useLayoutEffect, useRef, useState } from "react";
import { parseCredentialMentions, type CredentialMentionPart } from "./helpers";
import {
  editorSelectionOffset,
  readMentionEditor,
  setEditorCaret,
} from "./editorHelpers";

interface BadgeMount {
  element: HTMLSpanElement;
  mention: CredentialMentionPart;
}

export function useCredentialMentionEditor(
  value: string,
  onMultilineChange?: (multiline: boolean) => void,
) {
  const editorRef = useRef<HTMLDivElement>(null);
  const [badges, setBadges] = useState<BadgeMount[]>([]);
  const onMultilineRef = useRef(onMultilineChange);
  onMultilineRef.current = onMultilineChange;

  useLayoutEffect(() => {
    const editor = editorRef.current;
    if (!editor || readMentionEditor(editor) === value) return;
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
  }, [value]);

  useLayoutEffect(() => {
    if (editorRef.current) onMultilineRef.current?.(true);
  }, [value]);

  return { editorRef, badges };
}
