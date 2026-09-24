import type { MentionInput } from "../ChatInput/useChatMentions";

export function readMentionEditor(node: Node): string {
  if (node instanceof HTMLElement && node.dataset.credentialMention)
    return node.dataset.credentialMention;
  if (node.nodeType === Node.TEXT_NODE) return node.textContent ?? "";
  if (node.nodeName === "BR") return "\n";
  return Array.from(node.childNodes)
    .map((child, index) => {
      const separator =
        index > 0 && (child.nodeName === "DIV" || child.nodeName === "P")
          ? "\n"
          : "";
      return separator + readMentionEditor(child);
    })
    .join("");
}

export function editorSelectionOffset(root: HTMLElement) {
  const selection = window.getSelection();
  if (!selection?.rangeCount || !root.contains(selection.anchorNode))
    return readMentionEditor(root).length;
  const range = selection.getRangeAt(0).cloneRange();
  range.selectNodeContents(root);
  range.setEnd(selection.anchorNode!, selection.anchorOffset);
  return readMentionEditor(range.cloneContents()).length;
}

export function setEditorCaret(root: HTMLElement, offset: number) {
  const range = document.createRange();
  let remaining = Math.max(0, offset);
  function visit(node: Node): boolean {
    const token =
      node instanceof HTMLElement ? node.dataset.credentialMention : undefined;
    if (token) {
      if (remaining <= token.length) {
        if (remaining === 0) range.setStartBefore(node);
        else range.setStartAfter(node);
        return true;
      }
      remaining -= token.length;
      return false;
    }
    if (node.nodeType === Node.TEXT_NODE) {
      const length = node.textContent?.length ?? 0;
      if (remaining <= length) {
        range.setStart(node, remaining);
        return true;
      }
      remaining -= length;
      return false;
    }
    for (const child of Array.from(node.childNodes))
      if (visit(child)) return true;
    return false;
  }
  if (!visit(root)) {
    range.selectNodeContents(root);
    range.collapse(false);
  } else range.collapse(true);
  const selection = window.getSelection();
  selection?.removeAllRanges();
  selection?.addRange(range);
}

export function mentionInputFor(root: HTMLElement): MentionInput {
  return {
    get value() {
      return readMentionEditor(root);
    },
    get selectionStart() {
      return editorSelectionOffset(root);
    },
    setSelectionRange(start: number) {
      setEditorCaret(root, start);
    },
  };
}
